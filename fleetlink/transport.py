"""Transports for fleetlink: a zenoh one for real deployments and an in-memory
link emulator (bandwidth / latency / loss / bounded queues, virtual clock) for
tests and benchmarks.

Stream classes (API.md §8).  **Every class is congestion-control DROP**: a
must-deliver (BLOCK) message on a congested radio link stalls the whole zenoh
transport and, after ``wait_before_close``, closes it (zenoh #1876).  Delivery
guarantees live in the application instead: C0 requests are retried until
acked, C2 gaps are repaired with ``resend`` from the agent's ring buffer (and a
full resync as the last resort), C3 tiles through the manifest.

* ``C0`` control  - INTERACTIVE_HIGH, DROP, express, reliable
* ``C1`` state    - INTERACTIVE_LOW, DROP
* ``C2`` graph    - DATA_HIGH, DROP, reliable
* ``C3`` mesh     - DATA_LOW, DROP, best effort
* ``C4`` media    - DATA, DROP, request/reply only (on-demand camera frames)
* ``C5`` video    - DATA_LOW, DROP, best effort (live camera video, API.md §12)
* ``C6`` store    - BACKGROUND, DROP, reliable (live image-store mirror, API.md §13;
  lowest priority, gaps repaired by the base's watermark requests)

Request/reply (``declare_queryable`` / ``query``) carries C4: the base asks,
the robot answers once.  Replies travel above C3 (mesh / grid) in priority,
because an operator is waiting for them, and are rate-limited by the agent's
own C4 budget rather than by the scheduler.
"""

from __future__ import annotations

import heapq
import itertools
import json
import random
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

CLASSES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
# strict transmit order on the link (zenoh priority order): C4 sits between
# graph (C2) and mesh (C3); video (C5) shares DATA_LOW with C3; the image-store
# mirror (C6) is BACKGROUND and goes after everything
PRIORITY_ORDER = ("C0", "C1", "C2", "C4", "C3", "C5", "C6")
PRIORITY = {c: i for i, c in enumerate(PRIORITY_ORDER)}  # lower = more important

Callback = Callable[[str, bytes], None]
# queryable handler: (key, params) -> reply bytes, or None for no reply
QueryHandler = Callable[[str, dict], "bytes | None"]
# query callback: reply bytes, or None on timeout / no replier
ReplyCallback = Callable[["bytes | None"], None]


def encode_params(params: dict | None) -> str:
    """``{"w": 640, "q": 70}`` → ``"w=640;q=70"`` (zenoh selector parameters)."""
    return ";".join(f"{k}={v}" for k, v in (params or {}).items())


def decode_params(text: str) -> dict:
    out = {}
    for part in (text or "").replace("&", ";").split(";"):
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k] = v
    return out


def key_matches(expr: str, key: str) -> bool:
    """Zenoh-style key-expression match supporting ``*`` (one chunk) and ``**``."""
    e, k = expr.split("/"), key.split("/")

    def rec(i: int, j: int) -> bool:
        if i == len(e):
            return j == len(k)
        if e[i] == "**":
            return any(rec(i + 1, jj) for jj in range(j, len(k) + 1))
        if j == len(k):
            return False
        return (e[i] == "*" or e[i] == k[j]) and rec(i + 1, j + 1)

    return rec(0, 0)


def split_endpoints(connect: str | list[str] | tuple | None) -> list[str]:
    """``"tcp/a:7450,tcp/b:7450"`` or a list → ``["tcp/a:7450", "tcp/b:7450"]``."""
    if connect is None:
        return []
    items = [connect] if isinstance(connect, str) else list(connect)
    return [e.strip() for item in items for e in str(item).split(",") if e.strip()]


class Transport:
    """Minimal pub/sub interface used by the agent and the receiver."""

    def put(self, key: str, data: bytes, cls: str) -> None:
        raise NotImplementedError

    def subscribe(self, key_expr: str, callback: Callback):
        raise NotImplementedError

    def can_send(self, cls: str) -> bool:
        """False when the transport wants the scheduler to hold ``cls`` back."""
        return True

    def declare_queryable(self, key_expr: str, handler: QueryHandler):
        """Answer queries on ``key_expr``; ``handler(key, params)`` returns the reply
        bytes (or None for no reply).  It runs on the transport's thread."""
        raise NotImplementedError

    def query(
        self,
        key: str,
        params: dict | None,
        callback: ReplyCallback,
        timeout: float = 5.0,
    ) -> None:
        """Non-blocking request: ``callback(reply_bytes)`` once, or ``callback(None)``
        after ``timeout`` seconds without a reply."""
        raise NotImplementedError

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------
# Zenoh
# ----------------------------------------------------------------------------

# Zenoh 1.8 config keys, verified against ``zenoh.Config()`` of eclipse-zenoh
# 1.8.0 (it rejects unknown keys) and by opening sessions against a dead port
# and through a router restart:
#
# * client mode is required: a *peer* opens without a router, but a router
#   does not forward between peers that cannot reach each other directly;
# * a 1.8 client cannot open without a router: ``connect/timeout_ms: -1``
#   blocks ``zenoh.open`` forever and any other value raises "Unable to
#   connect".  So ``timeout_ms`` is 0 (fail fast) and :class:`ZenohTransport`
#   retries the open on a background thread with the ``connect/retry`` backoff;
# * once open, a client reconnects by itself after the router restarts
#   (``exit_on_failure: false``, the same ``connect/retry`` periods).
ZENOH_DEFAULTS = {
    "scouting/multicast/enabled": False,
    "connect/timeout_ms": 0,
    "connect/exit_on_failure": False,
    "connect/retry": {
        "period_init_ms": 500,
        # capped at 1 s, not 4: after a long radio dropout the backoff has reached
        # its cap, and C1 must be back within ~2 s of the link returning
        # (tests/test_fleet_harness.py stall test; 4 s missed it)
        "period_max_ms": 1000,
        "period_increase_factor": 2,
    },
    # Keep the backlog in zenoh's priority queues, where DROP and priorities act,
    # not in the kernel: an autotuned TCP send buffer grew to 2.6 MB on a
    # saturated 30 kB/s link and put C0 commands 24 s behind bulk C3 (head-of-line;
    # tests/test_fleet_harness.py throttle test).  16 kB (the kernel doubles it)
    # caps that at ~1 s on 30 kB/s and still allows ~300 kB/s at 100 ms RTT.
    "transport/link/tcp/so_sndbuf": 16384,
    # a batch being written cannot be pre-empted: 8 kB is 0.27 s at 30 kB/s
    # (64 kB, the default, is 2 s).  Negotiated: the smaller side wins.
    "transport/link/tx/batch_size": 8192,
    # ms before an unanswered endpoint (e.g. a dead radio) is given up for the next
    # one in the list; zenoh's default is 10 s
    "transport/unicast/open_timeout": 3000,
    # µs a DROP message waits for a free batch before it is dropped
    "transport/link/tx/queue/congestion_control/drop/wait_before_drop": 1000,
    "transport/link/tx/queue/congestion_control/drop/max_wait_before_drop_fragments": 50000,
    # batches per priority queue (zenoh default 2, maximum 16).  A message is
    # fragmented into batch_size pieces, and a DROP message whose fragments find no
    # free batch within the fragment wait is dropped whole (zenoh grows that wait
    # 1, 2, 4 ... ms per fragment: ~255 ms for 8 fragments).  With 2 x 8 kB, any
    # message over ~16 kB on a busy link (C2 snapshot chunks and deltas up to
    # max_msg_bytes, C3 tiles and grids, C4 JPEG replies, C5 key frames) was dropped
    # every time, and so was its resend.  16 x 8 kB = 128 kB per data priority
    # holds the largest of them whole.  Commands are unaffected: C0 / C1 have
    # their own (strictly higher) priority queues, and the kernel buffer stays small.
    "transport/link/tx/queue/size/data_high": 16,  # C2
    "transport/link/tx/queue/size/data": 16,  # C4 replies
    "transport/link/tx/queue/size/data_low": 16,  # C3, C5
    "transport/link/tx/queue/size/background": 16,  # C6 store frames (~40-130 kB)
}


class _Pending:
    """Handle of a subscriber / queryable declared before the session opened."""

    def __init__(self, kind: str, key_expr: str, fn):
        self.kind, self.key_expr, self.fn = kind, key_expr, fn
        self.handle = None

    def undeclare(self) -> None:
        if self.handle is not None:
            self.handle.undeclare()


class _PubRef:
    """A cached zenoh publisher and the puts currently using it."""

    __slots__ = ("pub", "users", "retired")

    def __init__(self, pub):
        self.pub, self.users, self.retired = pub, 0, False


class ZenohTransport(Transport):
    """eclipse-zenoh 1.x transport (client mode towards the configured routers).

    ``connect`` is one endpoint, a list, or a comma-separated list.
    ``zenoh_overrides`` maps config paths (``"transport/link/tx/lease"``) to
    values, applied after :data:`ZENOH_DEFAULTS`.

    The constructor never fails because no router is up: with
    ``wait_for_router`` (default) a failed open is retried on a background
    thread forever.  Until then puts are dropped (``unopened_drops``), queries
    answer None, and subscribers / queryables are declared once it opens
    (:attr:`opened` is set then).

    Puts are synchronous and never block for long (all classes DROP, the wait
    for a free batch is ``wait_before_drop``).  ``can_send(cls)`` is a *local*
    in-flight counter: False only while ``max_inflight`` puts of that class are
    still inside zenoh on other threads.  It does not see the link; real
    backpressure comes from the receiver's ``fb`` feedback (scheduler AIMD).

    Publishers are cached per (key, class), least recently used first out at
    ``max_publishers``; :meth:`undeclare` drops the ones of a key early.  A
    publisher evicted while a put on another thread is using it is undeclared
    when that put returns (puts never hold the lock while inside zenoh).
    """

    def __init__(
        self,
        connect: str | list[str] = "tcp/127.0.0.1:7447",
        mode: str = "client",
        session=None,
        zenoh_overrides: dict | None = None,
        max_publishers: int = 1024,
        max_inflight: int = 1,
        wait_for_router: bool = True,
    ):
        import zenoh

        self._z = zenoh
        self.endpoints = split_endpoints(connect)
        drop = zenoh.CongestionControl.DROP
        self._qos = {
            "C0": dict(
                priority=zenoh.Priority.INTERACTIVE_HIGH,
                congestion_control=drop,
                express=True,
                reliability=zenoh.Reliability.RELIABLE,
            ),
            "C1": dict(
                priority=zenoh.Priority.INTERACTIVE_LOW, congestion_control=drop
            ),
            "C2": dict(
                priority=zenoh.Priority.DATA_HIGH,
                congestion_control=drop,
                reliability=zenoh.Reliability.RELIABLE,
            ),
            "C3": dict(
                priority=zenoh.Priority.DATA_LOW,
                congestion_control=drop,
                reliability=zenoh.Reliability.BEST_EFFORT,
            ),
            "C4": dict(priority=zenoh.Priority.DATA, congestion_control=drop),
            "C5": dict(
                priority=zenoh.Priority.DATA_LOW,
                congestion_control=drop,
                reliability=zenoh.Reliability.BEST_EFFORT,
            ),
            "C6": dict(
                priority=zenoh.Priority.BACKGROUND,
                congestion_control=drop,
                reliability=zenoh.Reliability.RELIABLE,
            ),
        }
        self.max_publishers = int(max_publishers)
        self.max_inflight = int(max_inflight)
        self._pubs: OrderedDict[tuple[str, str], _PubRef] = OrderedDict()
        self._subs = []
        self._lock = threading.Lock()
        self._inflight = dict.fromkeys(CLASSES, 0)
        self.put_errors = 0
        self.unopened_drops = 0
        self.open_attempts = 0
        self.open_error: str | None = None
        self.opened = threading.Event()
        self._closed = False
        self._stop = threading.Event()
        self._opener: threading.Thread | None = None
        self.session = None
        self._own_session = session is None
        if session is not None:
            self.session = session
            self.opened.set()
            return
        self._conf_json = {
            "mode": mode,
            "connect/endpoints": self.endpoints,
            **ZENOH_DEFAULTS,
            **(zenoh_overrides or {}),
        }
        if not self._try_open():
            if not wait_for_router:
                raise RuntimeError(self.open_error)
            self._opener = threading.Thread(
                target=self._open_loop, daemon=True, name="fleetlink-zenoh-open"
            )
            self._opener.start()

    # -- session -------------------------------------------------------------

    def _config(self):
        conf = self._z.Config()
        for k, v in self._conf_json.items():
            conf.insert_json5(k, json.dumps(v))
        return conf

    def _try_open(self) -> bool:
        self.open_attempts += 1
        try:
            session = self._z.open(self._config())
        except Exception as exc:  # no router (yet)
            self.open_error = f"{type(exc).__name__}: {exc}"
            return False
        with self._lock:
            if self._closed:
                session.close()
                return False
            self.session = session
            pending = [s for s in self._subs if isinstance(s, _Pending)]
        for p in pending:
            self._declare(p)
        self.open_error = None
        self.opened.set()
        return True

    def _open_loop(self) -> None:
        retry = self._conf_json.get("connect/retry") or {}
        delay = float(retry.get("period_init_ms", 500)) / 1000.0
        dmax = float(retry.get("period_max_ms", 4000)) / 1000.0
        factor = float(retry.get("period_increase_factor", 2))
        while not self._stop.wait(delay):
            if self._try_open():
                return
            delay = min(dmax, delay * factor)

    def _declare(self, p: _Pending) -> None:
        if p.kind == "sub":
            p.handle = self.session.declare_subscriber(p.key_expr, p.fn)
        else:
            p.handle = self.session.declare_queryable(p.key_expr, p.fn)

    def _acquire(self, key: str, cls: str) -> "_PubRef":
        """The cached publisher of (key, cls), held (``users``) until :meth:`_release`.
        An LRU-evicted or undeclared publisher still held by a put on another thread
        is only undeclared when that put returns."""
        evicted = []
        with self._lock:
            ref = self._pubs.get((key, cls))
            if ref is None:
                ref = _PubRef(self.session.declare_publisher(key, **self._qos[cls]))
                self._pubs[(key, cls)] = ref
                while len(self._pubs) > self.max_publishers:
                    evicted.append(self._pubs.popitem(last=False)[1])
            else:
                self._pubs.move_to_end((key, cls))
            ref.users += 1
            evicted = [r for r in evicted if self._retire(r)]
        for r in evicted:
            self._undeclare_quietly(r.pub)
        return ref

    def _release(self, ref: "_PubRef") -> None:
        with self._lock:
            ref.users -= 1
            gone = ref.retired and ref.users == 0
        if gone:
            self._undeclare_quietly(ref.pub)

    @staticmethod
    def _retire(ref: "_PubRef") -> bool:
        """Lock held: out of the cache; True when nobody uses it (undeclare now)."""
        ref.retired = True
        return ref.users == 0

    @staticmethod
    def _undeclare_quietly(p) -> None:
        try:
            p.undeclare()
        except Exception:
            pass

    def undeclare(self, key: str) -> int:
        """Undeclare the cached publishers of ``key`` (all classes); returns how many
        (one still in a put on another thread goes when that put returns)."""
        with self._lock:
            gone = [k for k in self._pubs if k[0] == key]
            refs = [self._pubs.pop(k) for k in gone]
            now = [r for r in refs if self._retire(r)]
        for r in now:
            self._undeclare_quietly(r.pub)
        return len(refs)

    def publisher_count(self) -> int:
        with self._lock:
            return len(self._pubs)

    # -- pub/sub -------------------------------------------------------------

    def put(self, key: str, data: bytes, cls: str) -> None:
        if self._closed:
            return
        if self.session is None:
            self.unopened_drops += 1
            return
        with self._lock:
            self._inflight[cls] += 1
        ref = None
        try:
            ref = self._acquire(key, cls)
            ref.pub.put(data)
        except Exception:  # session closing underneath us: the message is lost
            self.put_errors += 1
        finally:
            if ref is not None:
                self._release(ref)
            with self._lock:
                self._inflight[cls] -= 1

    def can_send(self, cls: str) -> bool:
        return self._inflight.get(cls, 0) < self.max_inflight

    def _add(self, kind: str, key_expr: str, fn):
        with self._lock:
            if self.session is None:
                p = _Pending(kind, key_expr, fn)
                self._subs.append(p)
                return p
        if kind == "sub":
            h = self.session.declare_subscriber(key_expr, fn)
        else:
            h = self.session.declare_queryable(key_expr, fn)
        with self._lock:
            self._subs.append(h)
        return h

    def subscribe(self, key_expr: str, callback: Callback):
        def handler(sample):
            callback(str(sample.key_expr), sample.payload.to_bytes())

        return self._add("sub", key_expr, handler)

    def declare_queryable(self, key_expr: str, handler: QueryHandler):
        # replies inherit the query's QoS (C4 priority, DROP: set in ``query``);
        # zenoh 1.8 ignores (and warns about) QoS arguments on ``reply``
        def on_query(query):
            try:
                key = str(query.key_expr)
                data = handler(key, decode_params(str(query.parameters)))
                if data is not None:
                    query.reply(query.key_expr, bytes(data))
            except Exception:  # a bad request must never kill zenoh's thread
                pass

        return self._add("queryable", key_expr, on_query)

    def query(self, key, params, callback, timeout: float = 5.0) -> None:
        if self._closed or self.session is None:
            callback(None)
            return
        z = self._z
        done = threading.Lock()
        state = {"answered": False}

        def finish(data):
            with done:
                if state["answered"]:
                    return
                state["answered"] = True
            callback(data)

        def on_reply(reply):
            ok = reply.ok
            if ok is not None:
                finish(ok.payload.to_bytes())

        sel = key + ("?" + encode_params(params) if params else "")
        try:
            self.session.get(
                sel,
                z.handlers.Callback(on_reply, lambda: finish(None)),
                timeout=float(timeout),
                **self._qos["C4"],
            )
        except Exception:
            finish(None)

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._closed = True
            subs, self._subs = self._subs, []
            pubs = [r.pub for r in self._pubs.values()]
            self._pubs.clear()
        self._stop.set()
        if self._opener is not None:
            self._opener.join(timeout=5.0)
        for s in subs:
            self._undeclare_quietly(s)
        for p in pubs:
            self._undeclare_quietly(p)
        if self._own_session and self.session is not None:
            self.session.close()


# ----------------------------------------------------------------------------
# In-memory link emulator
# ----------------------------------------------------------------------------


class VirtualClock:
    """Manually advanced clock; call it to read the time."""

    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


@dataclass
class _Msg:
    key: str
    data: bytes
    cls: str
    remaining: int
    lost: bool = False
    extra_delay: float = 0.0
    kind: str = "put"  # put | query | reply
    qid: int = 0
    params: dict | None = None
    order: int = 0  # enqueue order (the tcp_hol FIFO)


@dataclass
class LinkStats:
    put_bytes: dict = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    delivered_bytes: dict = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    wire_bytes: dict = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    congestion_drops: dict = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    loss_drops: dict = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    retransmits: dict = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    closed_drops: dict = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))


class _Direction:
    """One direction of the link: per-priority tx queues, fragment-level strict
    priority (like zenoh's batching), serialisation at ``bandwidth`` and
    propagation ``latency``.

    Loss is applied per fragment.  Reliable classes retransmit the fragment
    (bandwidth is spent again and delivery is delayed by one RTT); best-effort
    classes lose the whole message.  Under congestion (class queue over
    ``queue_bytes``) DROP classes are dropped at ``put``; classes listed in the
    link's ``block_classes`` are accepted instead (a blocked put).  With
    ``backpressure`` ``can_send`` turns False while the queue is full.

    ``tcp_hol``: one byte-ordered FIFO for all classes (whole messages in put
    order, one shared ``queue_bytes``), like everything sitting in a TCP send
    buffer: a big message delays everything behind it, whatever its priority.
    """

    def __init__(self, link: LinkEmulator, rng: random.Random):
        self.link = link
        self.rng = rng
        self.queues = {c: deque() for c in CLASSES}
        self.queued = dict.fromkeys(CLASSES, 0)
        self.tx_time = 0.0
        self.inflight: list = []
        self.last_delivery = dict.fromkeys(CLASSES, 0.0)
        self.order = itertools.count()
        self.fifo = itertools.count()
        self.stats = LinkStats()
        self.peer: MemoryTransport | None = None

    def busy(self) -> bool:
        return any(self.queues.values())

    def put(self, key: str, data: bytes, cls: str, now: float, **extra) -> bool:
        n = len(data)
        self.stats.put_bytes[cls] += n
        lk = self.link
        if lk.is_closed(now):
            self.stats.closed_drops[cls] += 1
            return False
        if cls not in lk.block_classes and self._queued(cls) + n > lk.queue_bytes:
            self.stats.congestion_drops[cls] += 1
            return False
        if not self.busy():
            self.tx_time = max(self.tx_time, now)
        self.queues[cls].append(_Msg(key, data, cls, n, order=next(self.fifo), **extra))
        self.queued[cls] += n
        return True

    def _queued(self, cls: str) -> int:
        return sum(self.queued.values()) if self.link.tcp_hol else self.queued[cls]

    def can_send(self, cls: str) -> bool:
        lk = self.link
        if not lk.backpressure and cls not in lk.block_classes:
            return True  # like zenoh DROP: the sender cannot see the queue
        return self._queued(cls) < lk.queue_bytes

    def reset(self) -> None:
        """Connection closed: everything queued or in flight is lost."""
        for c in CLASSES:
            self.stats.closed_drops[c] += len(self.queues[c])
            self.queues[c].clear()
            self.queued[c] = 0
        for _, _, msg in self.inflight:
            self.stats.closed_drops[msg.cls] += 1
        self.inflight.clear()

    def advance(self, t: float) -> list:
        """Transmit until ``t``; returns messages whose delivery time is <= t."""
        while self.tx_time < t and self.busy():
            self.send_fragment()
        if not self.busy():
            self.tx_time = max(self.tx_time, t)
        return self.deliver(t)

    def send_fragment(self, airtime: float = 1.0) -> float:
        """Put the next fragment on the air (strict priority, or FIFO with
        ``tcp_hol``) at ``tx_time``; returns the seconds it took.  ``airtime``
        scales that time (e.g. a multi-hop relay spends the channel per hop);
        :class:`SharedChannel` drives several directions through this."""
        lk = self.link
        if lk.tcp_hol:
            cls = min(
                (c for c in CLASSES if self.queues[c]),
                key=lambda c: self.queues[c][0].order,
            )
        else:
            cls = next(c for c in PRIORITY_ORDER if self.queues[c])
        msg = self.queues[cls][0]
        frag = min(lk.mtu, msg.remaining)
        dt = frag / lk.bandwidth * airtime if lk.bandwidth else 0.0
        self.tx_time += dt
        self.stats.wire_bytes[cls] += frag
        if lk.loss and self.rng.random() < lk.loss:
            if lk.reliable.get(cls, False):
                self.stats.retransmits[cls] += 1
                msg.extra_delay += 2.0 * lk.latency
                return dt  # resend the same fragment
            msg.lost = True
        msg.remaining -= frag
        if msg.remaining > 0:
            return dt
        self.queues[cls].popleft()
        self.queued[cls] -= len(msg.data)
        if msg.lost:
            self.stats.loss_drops[cls] += 1
            return dt
        when = self.tx_time + lk.latency + msg.extra_delay
        fifo = "all" if lk.tcp_hol else cls  # one stream: one delivery order
        when = max(when, self.last_delivery.get(fifo, 0.0))  # FIFO
        self.last_delivery[fifo] = when
        heapq.heappush(self.inflight, (when, next(self.order), msg))
        return dt

    def deliver(self, t: float) -> list:
        """Messages whose delivery time is <= ``t``."""
        out = []
        while self.inflight and self.inflight[0][0] <= t:
            _, _, msg = heapq.heappop(self.inflight)
            self.stats.delivered_bytes[msg.cls] += len(msg.data)
            out.append(msg)
        return out


class LinkEmulator:
    """Two :class:`MemoryTransport` endpoints (``.robot`` and ``.base``) joined by
    a simulated radio link.  Deterministic: driven by a :class:`VirtualClock`
    and a seeded RNG; call :meth:`advance` after moving the clock.

    ``bandwidth`` is bytes/s per direction (None = infinite).

    ``block_classes`` lists classes that model zenoh BLOCK (accepted over the
    queue limit); the default is none, matching :class:`ZenohTransport`.
    ``backpressure=False`` makes ``can_send`` always True for DROP classes (a
    real zenoh sender cannot see its queue), so only feedback limits the rate.
    ``tcp_hol`` replaces the per-priority queues with one FIFO (see
    :class:`_Direction`).  :meth:`close_transport` emulates a transport close.
    """

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        bandwidth: float | None = 250_000.0,
        latency: float = 0.15,
        loss: float = 0.0,
        queue_bytes: int = 256 * 1024,
        mtu: int = 8192,
        seed: int = 0,
        reliable: dict | None = None,
        block_classes: tuple | list = (),
        backpressure: bool = True,
        tcp_hol: bool = False,
    ):
        self.clock = clock or VirtualClock()
        self.bandwidth = bandwidth
        self.latency = latency
        self.loss = loss
        self.queue_bytes = queue_bytes
        self.mtu = mtu
        self.block_classes = frozenset(block_classes)
        self.backpressure = backpressure
        self.tcp_hol = tcp_hol
        self.closed_until: float | None = None
        self.closes = 0
        self.reliable = reliable or {
            "C0": True,
            "C1": True,
            "C2": True,
            "C3": False,
            "C4": True,
            "C5": False,
            "C6": True,
        }
        self._lock = threading.RLock()
        rng = random.Random(seed)
        self.robot = MemoryTransport(
            self, _Direction(self, random.Random(rng.random()))
        )
        self.base = MemoryTransport(self, _Direction(self, random.Random(rng.random())))
        self.robot.tx.peer = self.base
        self.base.tx.peer = self.robot

    def advance(self, t: float | None = None) -> None:
        """Move both directions forward to ``t`` (default: the clock) and deliver."""
        t = self.clock() if t is None else t
        with self._lock:
            # deliver in time order across both directions; callbacks may put
            for _ in range(1000):
                batch = []
                for d in (self.robot.tx, self.base.tx):
                    batch.extend((m, d.peer) for m in d.advance(t))
                if not batch:
                    break
                for msg, peer in batch:
                    peer._deliver_msg(msg)
            for end in (self.robot, self.base):
                end._expire_queries(t)

    def stats(self) -> dict:
        return {"up": self.robot.tx.stats, "down": self.base.tx.stats}

    def is_closed(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        return self.closed_until is not None and now < self.closed_until

    def close_transport(self, duration_s: float) -> None:
        """Emulate a zenoh transport close (lease expiry / #1876): everything
        queued or in flight in both directions is lost, and every put (and query)
        is dropped for ``duration_s``; then the link reopens by itself."""
        with self._lock:
            self.closes += 1
            self.closed_until = self.clock() + float(duration_s)
            for d in (self.robot.tx, self.base.tx):
                d.reset()


class SharedChannel:
    """Several :class:`LinkEmulator` links on ONE radio channel (a MANET shares
    airtime among every node, both directions).  Each link keeps its own queues,
    priorities, latency and loss; the channel puts one fragment on the air at a
    time, round-robin over the busy directions of all links (roughly what TDMA /
    CSMA fairness gives each radio), each fragment taking ``size / rate`` seconds
    times the link's ``airtime`` factor (2 for a robot one relay hop away, ...).

    Give every link ``bandwidth`` = the channel rate (bytes/s) and drive time with
    :meth:`advance` instead of the links' own ``advance``.
    """

    def __init__(self, links: list, clock: Callable[[], float], airtime: list | None = None):
        self.links = list(links)
        self.clock = clock
        self.airtime = list(airtime) if airtime else [1.0] * len(self.links)
        self.air_t = clock()
        self._rr = 0
        self.busy_s = 0.0  # seconds the channel was on the air

    def _dirs(self):
        for lk, f in zip(self.links, self.airtime):
            yield lk, lk.robot.tx, f
            yield lk, lk.base.tx, f

    def advance(self, t: float | None = None) -> None:
        t = self.clock() if t is None else t
        dirs = list(self._dirs())
        while self.air_t < t:
            busy = [x for x in dirs if x[1].busy() and not x[0].is_closed(self.air_t)]
            if not busy:
                break
            lk, d, f = busy[self._rr % len(busy)]
            self._rr += 1
            with lk._lock:
                d.tx_time = self.air_t
                dt = d.send_fragment(f)
            self.air_t += dt
            self.busy_s += dt
        self.air_t = max(self.air_t, t)
        for lk in self.links:
            with lk._lock:
                for _ in range(1000):
                    batch = []
                    for d in (lk.robot.tx, lk.base.tx):
                        batch.extend((m, d.peer) for m in d.deliver(t))
                    if not batch:
                        break
                    for msg, peer in batch:
                        peer._deliver_msg(msg)
                for end in (lk.robot, lk.base):
                    end._expire_queries(t)


class MemoryTransport(Transport):
    """One end of a :class:`LinkEmulator`."""

    def __init__(self, link: LinkEmulator, tx: _Direction):
        self.link = link
        self.tx = tx
        self._subs: list[tuple[str, Callback]] = []
        self._queryables: list[tuple[str, QueryHandler]] = []
        self._queries: dict[int, tuple[float, ReplyCallback]] = {}  # qid -> deadline
        self._qids = itertools.count(1)
        self._closed = False

    def put(self, key: str, data: bytes, cls: str) -> None:
        if self._closed:
            return
        with self.link._lock:
            self.tx.put(key, bytes(data), cls, self.link.clock())

    def can_send(self, cls: str) -> bool:
        return self.tx.can_send(cls)

    def subscribe(self, key_expr: str, callback: Callback):
        handle = (key_expr, callback)
        self._subs.append(handle)
        return handle

    def unsubscribe(self, handle) -> None:
        if handle in self._subs:
            self._subs.remove(handle)

    def declare_queryable(self, key_expr: str, handler: QueryHandler):
        handle = (key_expr, handler)
        self._queryables.append(handle)
        return handle

    def query(self, key, params, callback, timeout: float = 5.0) -> None:
        """The request crosses the link as a small C4 message; the reply comes back
        as C4 on the peer's direction (bandwidth, latency and loss apply).  Expiry
        is checked in :meth:`LinkEmulator.advance`, so it follows the clock."""
        if self._closed:
            callback(None)
            return
        with self.link._lock:
            qid = next(self._qids)
            now = self.link.clock()
            self._queries[qid] = (now + timeout, callback)
            req = (key + "?" + encode_params(params)).encode()
            self.tx.put(key, req, "C4", now, kind="query", qid=qid, params=params)

    def _expire_queries(self, t: float) -> None:
        for qid, (deadline, cb) in list(self._queries.items()):
            if deadline <= t:
                del self._queries[qid]
                cb(None)

    def _deliver_msg(self, msg: _Msg) -> None:
        if self._closed:
            return
        if msg.kind == "put":
            self._deliver(msg.key, msg.data)
        elif msg.kind == "query":
            for expr, handler in list(self._queryables):
                if key_matches(expr, msg.key):
                    try:
                        data = handler(msg.key, dict(msg.params or {}))
                    except Exception:
                        data = None
                    if data is not None:
                        self.tx.put(
                            msg.key,
                            bytes(data),
                            "C4",
                            self.link.clock(),
                            kind="reply",
                            qid=msg.qid,
                        )
                    break
        elif msg.kind == "reply":
            pending = self._queries.pop(msg.qid, None)
            if pending is not None:
                pending[1](msg.data)

    def _deliver(self, key: str, data: bytes) -> None:
        if self._closed:
            return
        for expr, cb in list(self._subs):
            if key_matches(expr, key):
                cb(key, data)

    def close(self) -> None:
        self._closed = True
        self._subs.clear()
        self._queryables.clear()
        for _, cb in self._queries.values():
            cb(None)
        self._queries.clear()

    def reopen(self) -> None:
        """Allow a new agent to reuse this endpoint after ``close`` (restart tests)."""
        self._closed = False
