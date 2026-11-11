"""Generic ROS relay over fleetlink: the pure (ROS-free) half.

The robot's ROS graph stays on the robot; the topics, services and TF frames that
must cross the radio are listed in ``fleetlink/config/relay.yaml`` and carried by
fleetlink.  :class:`RelayConfig` loads and validates that table, and
:class:`RelayCore` is one end of one robot<->base link: the robot runs one
(``side="robot"``, sends ``up``), the base bridge runs one per robot
(``side="base"``, sends ``down``).  ``relay_ros.py`` wires them to rclpy.

Wire (all keys under ``adt4/<robot>/ros/``, every message a
``codec.pack(codec.envelope(...))`` with ``t`` = the time the message was first
created, in the sender's clock):

* ``up/<topic>`` / ``down/<topic>`` (stream ``ros/up`` | ``ros/down``)
  ``{topic, type, th, mode, tseq, to, data, id?, lo?}``: ``data`` is the CDR
  (``rclpy.serialization``) bytes, ``th`` the type hash (RIHS01), ``tseq`` a
  per-topic counter, ``to`` the receiver epoch the sender believes is current
  (None before first contact), ``id`` for acked modes, ``lo`` (reliable) the
  oldest tseq the sender still retries.
* ``ack/up`` (sent by the robot) / ``ack/down`` (sent by the base), stream
  ``ros/ack``: ``{op: "ack", id, topic, ok, err?, subs?}`` (``subs`` = local
  subscribers of the republished topic) and once a second
  ``{op: "hb", seq, t, echo_t, held, cfg, platform}`` (heartbeat: liveness, RTT,
  clock offset, loss, config hash).
* ``srv/req`` / ``srv/rep`` (stream ``ros/srv``), both directions, told apart by
  ``side``: ``{op: "req", side, id, service, type, th, to, data}`` /
  ``{op: "rep", side, id, ok, err?, data?}``.  Requests are retried every
  ``retry_s`` until a reply or ``timeout_s``; the server dedupes by ``id``
  (in-progress requests are ignored, finished ones answered from a cache).

Delivery modes (per topic):

=================  ==========================================================
``reliable``       in order; queue cap 32 (oldest dropped); retried every 1 s,
                   30 tries; the receiver acks, dedupes and applies in tseq
                   order (held messages wait for gaps; ``lo`` skips what the
                   sender gave up on)
``latest_reliable`` one pending slot: a newer message replaces the pending
                   retry; the receiver applies only tseq above the last applied
``latest``         rate-limited (``rate_hz``; the newest pending value is sent
                   at the next slot), fire and forget; stale tseq dropped
=================  ==========================================================

**Large messages.**  In the acked modes a message larger than ``chunk_bytes``
(16 kB) is sent in parts: ``{..., id, part, parts, data}`` with the same
``id`` / ``tseq``, at most ``chunk_window`` (6) parts unacknowledged at a time (so
a few hundred kB never overflows zenoh's 128 kB per-priority queue, which drops a
message that does not fit whole).  The receiver answers each part with
``{op: "pack", id, topic, part}`` and, once it has every part, reassembles the
message and handles it as one (type, ttl, order, the usual ``ack``).  The sender
resends only the parts not acked: after a timer set from the pace of part acks
(at least ``retry_s``, doubled on a timeout without progress, at most 8 s), or at
once when a part sent after it is acked first (fast retransmit).  The give-up
budget (``max_tries``) counts timeouts without progress.  ``latest`` (fire and
forget) is never split.

``ttl_s`` is checked on both ends: the sender stops retrying, the receiver
rejects (``ok: false, err: "expired"``) a message older than that, measured
against its envelope ``t`` corrected by the heartbeat clock offset.
``ttl_strict`` (latest entries, e.g. teleop velocity) fails closed: no
round-trip offset yet means expired, and a latest value that waited past its
ttl for a rate slot is dropped on the sender.

**Epochs / no replay.**  Each end has a random epoch.  When the peer's epoch
*changes* (it restarted), pending messages of entries without
``replay_on_resync`` are dropped, entries with it re-address their pending
messages to the new peer (or, with nothing pending, a ``latest`` /
``latest_reliable`` entry re-sends its last value; a ``reliable`` one does not
re-send what the old peer acked), and messages still addressed to the old epoch
are rejected by the new peer ("stale epoch", a nack that never cancels a pending
message).  So a rebooted robot never re-runs a stale plan.  The receiver-side
TTL needs the clock offset: before the first heartbeat round trip it uses the
last heartbeat's one-way estimate, and before any heartbeat it is skipped.

**Loop prevention.**  :class:`RelayConfig` rejects any topic / service
configured in both directions, and each end only publishes its inbound
direction.  Everything is congestion-control DROP (``ZenohTransport``).
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from fleetlink import codec
from fleetlink.transport import CLASSES, Transport

MODES = ("reliable", "latest_reliable", "latest")
DIRS = ("up", "down")
SIDES = {"robot": "up", "base": "down"}  # side -> direction it sends
DEFAULT_QOS = {"reliability": "reliable", "durability": "volatile", "depth": 10}
TF_PREFIX = "/_tf"  # wire names of composed transforms: /_tf/<parent>/<child>
TF_STATIC = "/_tf_static"

_TOPIC_KEYS = {
    "topic",
    "type",
    "dir",
    "mode",
    "cls",
    "rate_hz",
    "ttl_s",
    "ttl_strict",
    "replay_on_resync",
    "qos",
    "platform",
}
_TF_KEYS = {
    "parent",
    "child",
    "static",
    "frames_prefix",
    "rate_hz",
    "ttl_s",
    "mode",
    "cls",
    "dir",
    "replay_on_resync",
    "platform",
}
_SRV_KEYS = {
    "service",
    "base_name",
    "type",
    "dir",
    "cls",
    "ttl_s",
    "timeout_s",
    "platform",
}


class RelayConfigError(ValueError):
    pass


@dataclass
class Entry:
    """One relayed topic (or composed TF) after templating."""

    name: str  # wire name: the ROS topic, or /_tf/<parent>/<child>, /_tf_static
    topic: str  # ROS topic on both ends (/tf, /tf_static for TF entries)
    type: str
    dir: str
    mode: str
    cls: str
    rate_hz: float | None = None
    ttl_s: float | None = None
    ttl_strict: bool = False
    replay_on_resync: bool = False
    qos: dict = field(default_factory=lambda: dict(DEFAULT_QOS))
    platform: tuple | None = None
    kind: str = "topic"  # topic | tf | tf_static
    parent: str = ""
    child: str = ""
    frames_prefix: str = ""


@dataclass
class ServiceEntry:
    name: str  # robot-side service name; also the wire name
    base_name: str  # base-side service name (proxy server or real server)
    type: str
    dir: str  # direction of the REQUEST
    cls: str = "C0"
    ttl_s: float | None = None
    timeout_s: float = 10.0
    platform: tuple | None = None


@dataclass
class Resolved:
    """The config for one robot: wire name -> entry."""

    topics: dict[str, Entry]
    services: dict[str, ServiceEntry]

    def out_topics(self, side: str) -> list[Entry]:
        return [e for e in self.topics.values() if e.dir == SIDES[side]]

    def in_topics(self, side: str) -> list[Entry]:
        return [e for e in self.topics.values() if e.dir != SIDES[side]]


def config_hash(raw: dict) -> str:
    text = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _platforms(v) -> tuple | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = [v]
    return tuple(str(p).lower() for p in v)


def _overlap(a: tuple | None, b: tuple | None) -> bool:
    return a is None or b is None or bool(set(a) & set(b))


def _platform_ok(entry_platforms: tuple | None, platform: str | None) -> bool:
    return entry_platforms is None or (
        platform is not None and platform.lower() in entry_platforms
    )


def _num(v, what: str, allow_none=True) -> float | None:
    if v is None and allow_none:
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        raise RelayConfigError(f"{what}: not a number: {v!r}") from None
    if out < 0:
        raise RelayConfigError(f"{what}: negative: {v!r}")
    return out


class RelayConfig:
    """``relay.yaml``: ``version``, ``topics``, ``tf``, ``services`` (schema in the
    file header).  Raises :class:`RelayConfigError` on an unknown key / mode /
    direction / class, a topic or service configured in both directions, or two
    same-direction entries for one name whose platforms overlap."""

    def __init__(self, raw: dict | None, source: str = ""):
        self.raw = raw or {}
        self.source = source
        self.hash = config_hash(self.raw)
        self.topics: list[dict] = []
        self.tf: list[dict] = []
        self.services: list[dict] = []
        self._validate()

    @classmethod
    def load(cls, path) -> "RelayConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw, str(path))

    # -- validation -----------------------------------------------------------

    def _err(self, msg: str):
        where = f"{self.source}: " if self.source else ""
        return RelayConfigError(where + msg)

    def _validate(self) -> None:
        raw = self.raw
        if not isinstance(raw, dict):
            raise self._err("top level must be a mapping")
        unknown = set(raw) - {"version", "topics", "tf", "services"}
        if unknown:
            raise self._err(f"unknown top-level keys {sorted(unknown)}")
        if int(raw.get("version", 1)) != 1:
            raise self._err(f"unsupported version {raw.get('version')!r}")
        for i, t in enumerate(raw.get("topics") or []):
            self.topics.append(self._check_topic(t, f"topics[{i}]"))
        for i, t in enumerate(raw.get("tf") or []):
            self.tf.append(self._check_tf(t, f"tf[{i}]"))
        for i, s in enumerate(raw.get("services") or []):
            self.services.append(self._check_srv(s, f"services[{i}]"))
        self._check_conflicts(
            [(t["topic"], t["dir"], t["platform"]) for t in self.topics], "topic"
        )
        self._check_conflicts(
            [(s["service"], s["dir"], s["platform"]) for s in self.services],
            "service",
        )

    def _check_common(self, d: dict, what: str, keys: set) -> None:
        if not isinstance(d, dict):
            raise self._err(f"{what}: must be a mapping")
        unknown = set(d) - keys
        if unknown:
            raise self._err(f"{what}: unknown keys {sorted(unknown)}")
        if d.get("dir") not in DIRS:
            raise self._err(f"{what}: dir must be up|down, got {d.get('dir')!r}")
        if "mode" in d and d["mode"] not in MODES:
            raise self._err(f"{what}: unknown mode {d['mode']!r} (one of {MODES})")
        if "cls" in d and d["cls"] not in CLASSES:
            raise self._err(f"{what}: unknown class {d['cls']!r}")

    def _check_topic(self, t: dict, what: str) -> dict:
        self._check_common(t, what, _TOPIC_KEYS)
        for k in ("topic", "type", "mode"):
            if not t.get(k):
                raise self._err(f"{what}: missing {k!r}")
        if not str(t["topic"]).startswith("/"):
            raise self._err(f"{what}: topic must be absolute: {t['topic']!r}")
        qos = dict(DEFAULT_QOS, **(t.get("qos") or {}))
        if qos["reliability"] not in ("reliable", "best_effort"):
            raise self._err(f"{what}: qos.reliability {qos['reliability']!r}")
        if qos["durability"] not in ("volatile", "transient_local"):
            raise self._err(f"{what}: qos.durability {qos['durability']!r}")
        mode = t["mode"]
        ttl_strict = bool(t.get("ttl_strict", False))
        if ttl_strict and t.get("ttl_s") is None:
            raise self._err(f"{what}: ttl_strict needs ttl_s")
        if ttl_strict and mode != "latest":
            # a strict ttl drops until the clock offset is measured; only a
            # stream that is resent continuously may lose its first messages
            raise self._err(f"{what}: ttl_strict is only for mode latest")
        return {
            "topic": str(t["topic"]),
            "type": str(t["type"]),
            "dir": t["dir"],
            "mode": mode,
            "cls": t.get("cls") or ("C1" if mode == "latest" else "C0"),
            "rate_hz": _num(t.get("rate_hz"), f"{what}.rate_hz"),
            "ttl_s": _num(t.get("ttl_s"), f"{what}.ttl_s"),
            "ttl_strict": ttl_strict,
            "replay_on_resync": bool(t.get("replay_on_resync", False)),
            "qos": {**qos, "depth": int(qos.get("depth") or 10)},
            "platform": _platforms(t.get("platform")),
        }

    def _check_tf(self, t: dict, what: str) -> dict:
        self._check_common(t, what, _TF_KEYS)
        if t["dir"] != "up":
            raise self._err(f"{what}: only dir: up transforms are supported")
        static = bool(t.get("static", False))
        if static:
            if not t.get("frames_prefix"):
                raise self._err(f"{what}: static tf needs frames_prefix")
        elif not t.get("parent") or not t.get("child"):
            raise self._err(f"{what}: needs parent and child (or static: true)")
        return {
            "static": static,
            "parent": str(t.get("parent") or ""),
            "child": str(t.get("child") or ""),
            "frames_prefix": str(t.get("frames_prefix") or ""),
            "rate_hz": _num(
                t.get("rate_hz", None if static else 5.0), f"{what}.rate_hz"
            ),
            "ttl_s": _num(t.get("ttl_s"), f"{what}.ttl_s"),
            "mode": t.get("mode") or ("latest_reliable" if static else "latest"),
            "cls": t.get("cls") or ("C0" if static else "C1"),
            "dir": "up",
            "replay_on_resync": bool(t.get("replay_on_resync", static)),
            "platform": _platforms(t.get("platform")),
        }

    def _check_srv(self, s: dict, what: str) -> dict:
        self._check_common(s, what, _SRV_KEYS)
        for k in ("service", "type"):
            if not s.get(k):
                raise self._err(f"{what}: missing {k!r}")
        if not str(s["service"]).startswith("/"):
            raise self._err(f"{what}: service must be absolute: {s['service']!r}")
        return {
            "service": str(s["service"]),
            "base_name": str(s.get("base_name") or s["service"]),
            "type": str(s["type"]),
            "dir": s["dir"],
            "cls": s.get("cls") or "C0",
            "ttl_s": _num(s.get("ttl_s"), f"{what}.ttl_s"),
            "timeout_s": _num(s.get("timeout_s", 10.0), f"{what}.timeout_s", False),
            "platform": _platforms(s.get("platform")),
        }

    def _check_conflicts(self, items, what: str, robot: str | None = None) -> None:
        seen: dict[str, list] = collections.defaultdict(list)
        for name, d, plat in items:
            for d2, plat2 in seen[name]:
                if d2 != d:
                    raise self._err(
                        f"{what} {name!r} is configured in both directions "
                        "(that would loop it back); pick one"
                    )
                if _overlap(plat, plat2):
                    raise self._err(
                        f"{what} {name!r} is configured twice for the same platform"
                    )
            seen[name].append((d, plat))

    # -- per robot ------------------------------------------------------------

    def resolve(
        self, robot: str, platform: str | None = None, base: str = ""
    ) -> Resolved:
        """Template ``{robot}`` / ``{base}`` and keep the entries for ``platform``
        (entries without ``platform`` are always kept)."""

        def fmt(s: str) -> str:
            return s.replace("{robot}", robot).replace("{base}", base or "base_station")

        topics: dict[str, Entry] = {}
        for t in self.topics:
            if not _platform_ok(t["platform"], platform):
                continue
            name = fmt(t["topic"])
            if name in topics:  # same pattern after templating, both directions
                raise self._err(f"topic {name!r} configured twice for {robot}")
            topics[name] = Entry(
                name=name,
                topic=name,
                type=t["type"],
                dir=t["dir"],
                mode=t["mode"],
                cls=t["cls"],
                rate_hz=t["rate_hz"],
                ttl_s=t["ttl_s"],
                ttl_strict=t["ttl_strict"],
                replay_on_resync=t["replay_on_resync"],
                qos=dict(t["qos"]),
                platform=t["platform"],
            )
        for t in self.tf:
            if not _platform_ok(t["platform"], platform):
                continue
            if t["static"]:
                e = Entry(
                    name=TF_STATIC,
                    topic="/tf_static",
                    type="tf2_msgs/msg/TFMessage",
                    kind="tf_static",
                    frames_prefix=fmt(t["frames_prefix"]),
                    qos={"reliability": "reliable", "durability": "transient_local", "depth": 1},
                    **{k: t[k] for k in ("dir", "mode", "cls", "rate_hz", "ttl_s", "replay_on_resync", "platform")},
                )  # fmt: skip
            else:
                parent, child = fmt(t["parent"]), fmt(t["child"])
                e = Entry(
                    name=f"{TF_PREFIX}/{parent.strip('/')}/{child.strip('/')}",
                    topic="/tf",
                    type="tf2_msgs/msg/TFMessage",
                    kind="tf",
                    parent=parent,
                    child=child,
                    qos={"reliability": "reliable", "durability": "volatile", "depth": 100},
                    **{k: t[k] for k in ("dir", "mode", "cls", "rate_hz", "ttl_s", "replay_on_resync", "platform")},
                )  # fmt: skip
            if e.name in topics:
                raise self._err(f"tf entry {e.name!r} configured twice for {robot}")
            topics[e.name] = e
        services: dict[str, ServiceEntry] = {}
        for s in self.services:
            if not _platform_ok(s["platform"], platform):
                continue
            name = fmt(s["service"])
            if name in services:
                raise self._err(f"service {name!r} configured twice for {robot}")
            services[name] = ServiceEntry(
                name=name,
                base_name=fmt(s["base_name"]),
                type=s["type"],
                dir=s["dir"],
                cls=s["cls"],
                ttl_s=s["ttl_s"],
                timeout_s=s["timeout_s"],
                platform=s["platform"],
            )
        return Resolved(topics, services)


# ----------------------------------------------------------------------------
# One end of one robot <-> base link
# ----------------------------------------------------------------------------


def _counters() -> dict:
    return dict.fromkeys(
        (
            "sent",  # first transmissions
            "resent",  # retries / replays
            "acked",
            "delivered",  # applied on this end (inbound)
            "dup",
            "expired",  # ttl (rx: rejected; tx: given up)
            "type_mismatch",
            "overflow",  # reliable queue cap
            "gave_up",  # retries exhausted
            "superseded",  # latest / latest_reliable replaced before sent / acked
            "rejected",  # negative acks received
            "stale_epoch",  # addressed to a previous incarnation of this end
            "dropped_restart",  # pending dropped because the peer restarted
            "parts_sent",  # parts of large messages (first sends and resends)
            "parts_rx",  # parts received
        ),
        0,
    )


@dataclass
class _Pending:
    id: str
    tseq: int
    data: bytes
    t: float  # created (sender clock)
    t_sent: float
    to: str | None
    tries: int = 1
    # large messages: number of parts, parts the receiver acked, part -> last sent
    parts: int = 1
    have: set = field(default_factory=set)
    part_sent: dict = field(default_factory=dict)
    # part resend timer: set from the measured pace of part acks (a part can take
    # longer than retry_s to serialise on a slow radio) and doubled on a timeout
    rto: float = 0.0
    ack_dt: float | None = None  # smoothed seconds between part acks
    last_pack_t: float | None = None
    send_n: int = 0  # parts put on the wire so far
    part_order: dict = field(default_factory=dict)  # part -> send_n of its last send


class _Out:
    """Sender state of one outbound entry."""

    def __init__(self, entry: Entry):
        self.entry = entry
        self.next_tseq = 0
        self.pending: collections.OrderedDict[str, _Pending] = collections.OrderedDict()
        self.last: tuple[bytes, float] | None = None  # (data, t) for replay
        self.latest_wait: tuple[bytes, float] | None = None  # rate-limited latest
        self.t_last_send = -1e18
        self.c = _counters()
        self.last_ack_t: float | None = None
        self.last_subs: int | None = None
        self.last_err: str | None = None


class _In:
    """Receiver state of one inbound entry (reset when the sender's epoch changes)."""

    def __init__(self):
        self.epoch: str | None = None
        self.last_applied: int | None = None  # latest / latest_reliable
        self.expected: int | None = None  # reliable: next tseq to apply
        self.held: dict[int, tuple] = {}  # reliable: tseq -> (data, meta)
        # large messages being reassembled: id -> {"n", "chunks", "t", "tseq"}
        self.partial: dict[str, dict] = {}
        # ids already reassembled (a late resent part is answered, not rebuilt)
        self.done: collections.OrderedDict[str, None] = collections.OrderedDict()
        self.c = _counters()
        self.last_rx: float | None = None

    def reset(self, epoch: str) -> None:
        self.epoch = epoch
        self.last_applied = None
        self.expected = None
        self.held.clear()
        self.partial.clear()
        self.done.clear()


@dataclass
class _Call:
    id: str
    service: ServiceEntry
    payload: dict
    t: float
    t_sent: float
    deadline: float
    callback: Callable


# on_message(entry, data, meta) -> local subscriber count (or None)
MessageHandler = Callable[[Entry, bytes, dict], "int | None"]
# on_service(service, request_bytes, reply) with reply(response_bytes | None, err | None)
ServiceHandler = Callable[[ServiceEntry, bytes, Callable], None]


class RelayCore:
    """One end (``side`` = ``robot`` | ``base``) of the relay for one robot.

    Thread-safe; transport callbacks may arrive on any thread.  Puts and user
    callbacks happen outside the internal lock.  Call :meth:`tick` regularly
    (10-20 Hz): retries, rate-limited ``latest`` values, heartbeats, service
    timeouts.  ``type_hash(type_str) -> str | None`` enables the type check.
    ``on_event(kind, info)`` reports ``ack`` / ``peer_restart`` / ``cfg_mismatch``.
    """

    def __init__(
        self,
        robot: str,
        side: str,
        transport: Transport,
        config: RelayConfig,
        *,
        platform: str | None = None,
        base: str = "",
        clock: Callable[[], float] = time.time,
        on_message: MessageHandler | None = None,
        on_service: ServiceHandler | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        type_hash: Callable[[str], "str | None"] | None = None,
        log: Callable[[str], None] | None = None,
        retry_s: float = 1.0,
        max_tries: int = 30,
        queue_cap: int = 32,
        hb_period: float = 1.0,
        connected_timeout: float = 3.0,
        srv_cache: int = 64,
        loss_window_s: float = 5.0,
        chunk_bytes: int = 16 * 1024,
        chunk_window: int = 6,
    ):
        if side not in SIDES:
            raise ValueError(f"side must be robot|base, got {side!r}")
        self.robot = robot
        self.side = side
        self.out_dir = SIDES[side]
        self.in_dir = "down" if self.out_dir == "up" else "up"
        self.transport = transport
        self.config = config
        self.platform = platform
        self.resolved = config.resolve(robot, platform, base)
        self.clock = clock
        self.on_message = on_message
        self.on_service = on_service
        self.on_event = on_event or (lambda kind, info: None)
        self._th = type_hash or (lambda t: None)
        self._th_cache: dict[str, str | None] = {}
        self.log = log or (lambda msg: None)
        self.retry_s = retry_s
        self.max_tries = max_tries
        self.queue_cap = queue_cap
        self.hb_period = hb_period
        self.connected_timeout = connected_timeout
        self.srv_cache = srv_cache
        self.loss_window_s = loss_window_s
        self.chunk_bytes = max(1024, int(chunk_bytes))
        self.chunk_window = max(1, int(chunk_window))

        self.epoch = secrets.token_hex(6)
        self.prefix = f"adt4/{robot}/ros"
        self._lock = threading.RLock()
        self._ids = itertools.count()
        self._seq = collections.defaultdict(itertools.count)  # stream -> seq
        self._outbox: list[tuple[str, bytes, str]] = []
        self._calls_out: list[tuple[Callable, tuple]] = []  # deferred user callbacks

        self.out = {e.name: _Out(e) for e in self.resolved.out_topics(side)}
        self.inn = {e.name: _In() for e in self.resolved.in_topics(side)}
        self._id_owner: dict[str, str] = {}  # pending id -> entry name
        # services: requests this side sends / serves
        self.srv_out = {
            n: s for n, s in self.resolved.services.items() if s.dir == self.out_dir
        }
        self.srv_in = {
            n: s for n, s in self.resolved.services.items() if s.dir == self.in_dir
        }
        self._calls: dict[str, _Call] = {}
        self._served: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self._serving: set[str] = set()
        self.srv_stats = {
            n: dict.fromkeys(("calls", "ok", "errors", "timeouts", "served", "dup"), 0)
            for n in self.resolved.services
        }

        # peer / link
        self.peer_epoch: str | None = None
        self._retired: set[str] = set()
        self.peer_restarts = 0
        self.peer_platform: str | None = None
        self.peer_cfg: str | None = None
        self._cfg_warned = False
        self.last_seen: float | None = None
        self.rtt: float | None = None
        self._offsets: collections.deque = collections.deque(maxlen=20)
        self.clock_offset: float | None = None  # peer clock - my clock
        self._peer_hb: tuple[float, float] | None = None  # (peer t, my rx time)
        self._t_hb = -1e18
        self._rx_seq: dict[str, int] = {}  # peer stream -> last seq
        self._loss_win = [0, 0]  # received, lost (current window)
        self._loss_t: float | None = None
        self.loss = 0.0
        self.unknown_topics = 0
        self.unknown_services = 0

        self._subs = [
            transport.subscribe(f"{self.prefix}/{self.in_dir}/**", self._on_data),
            transport.subscribe(f"{self.prefix}/ack/{self.in_dir}", self._on_ack),
            transport.subscribe(f"{self.prefix}/srv/req", self._on_srv_req),
            transport.subscribe(f"{self.prefix}/srv/rep", self._on_srv_rep),
        ]

    # ------------------------------------------------------------------ helpers

    def type_hash(self, type_str: str) -> str | None:
        if type_str not in self._th_cache:
            try:
                self._th_cache[type_str] = self._th(type_str)
            except Exception:
                self._th_cache[type_str] = None
        return self._th_cache[type_str]

    def _new_id(self) -> str:
        return f"{self.epoch[:6]}-{next(self._ids)}"

    def _put(self, key: str, stream: str, payload: dict, cls: str, t: float) -> None:
        """Lock held: queue one message for the outbox."""
        env = codec.envelope(
            self.robot, stream, next(self._seq[stream]), self.epoch, payload
        )
        env["t"] = t
        self._outbox.append((key, codec.pack(env), cls))

    def _flush(self) -> None:
        with self._lock:
            out, self._outbox = self._outbox, []
            calls, self._calls_out = self._calls_out, []
        for key, data, cls in out:
            try:
                self.transport.put(key, data, cls)
            except Exception as exc:  # a dead transport must not kill the caller
                self.log(f"relay put {key} failed: {exc}")
        for fn, args in calls:
            try:
                fn(*args)
            except Exception as exc:
                self.log(f"relay callback failed: {exc}")

    def _defer(self, fn, *args) -> None:
        if fn is not None:
            self._calls_out.append((fn, args))

    def _unpack(self, data: bytes) -> dict | None:
        try:
            env = codec.unpack(data)
            if env.get("epoch") == self.epoch or not isinstance(env.get("p"), dict):
                return None  # our own message (zenoh delivers locally too)
            return env
        except Exception:
            return None

    def _offset(self) -> float | None:
        """Peer clock - my clock: the heartbeat (NTP-style) estimate; before the
        first round trip, the one-way estimate of the last heartbeat (ages come
        out short by the one-way latency); None before any heartbeat."""
        if self.clock_offset is not None:
            return self.clock_offset
        if self._peer_hb is not None:
            return self._peer_hb[0] - self._peer_hb[1]
        return None

    def _expired(
        self, env: dict, ttl: float | None, now: float, strict: bool = False
    ) -> bool:
        """Receiver-side TTL.  Skipped while the clock offset is unknown (right
        after link-up and after a peer restart): the raw envelope ``t`` would be
        off by the clock skew, and the sender's own TTL still bounds its retries.
        ``strict`` (``ttl_strict``, e.g. teleop velocity) fails closed instead:
        until the round-trip offset is measured every message counts as expired,
        so a delayed motion command can never be applied."""
        if ttl is None:
            return False
        off = self.clock_offset if strict else self._offset()
        if off is None:
            return strict
        return now - (float(env.get("t", now)) - off) > ttl

    def _account_seq(self, env: dict) -> None:
        stream = env.get("stream", "")
        seq = int(env.get("seq", 0))
        last = self._rx_seq.get(stream)
        if last is not None and seq > last + 1:
            self._loss_win[1] += seq - last - 1
        if last is None or seq > last:
            self._rx_seq[stream] = seq
        self._loss_win[0] += 1

    def _peer_seen(self, env: dict, now: float) -> bool:
        """Lock held: track the peer's epoch.  False for a retired epoch."""
        epoch = env.get("epoch")
        if epoch in self._retired:
            return False
        if self.peer_epoch is None:
            self.peer_epoch = epoch
        elif epoch != self.peer_epoch:
            self._retired.add(self.peer_epoch)
            self.peer_epoch = epoch
            self.peer_restarts += 1
            self._rx_seq.clear()
            self._on_peer_restart(now)
        self.last_seen = now
        self._account_seq(env)
        return True

    def _on_peer_restart(self, now: float) -> None:
        """Lock held: the peer is a new incarnation.  Only ``replay_on_resync``
        entries re-send; everything else pending is dropped (no stale replays)."""
        self._offsets.clear()
        self.clock_offset = None
        self._peer_hb = None
        for o in self.out.values():
            e = o.entry
            if not e.replay_on_resync:
                o.c["dropped_restart"] += len(o.pending)
                for pid in o.pending:
                    self._id_owner.pop(pid, None)
                o.pending.clear()
                o.latest_wait = None
                continue
            if o.pending:
                for p in o.pending.values():
                    p.to = self.peer_epoch
                    p.tries = 0
                    p.t_sent = -1e18  # resent by the next tick
                    p.have.clear()  # a new peer has none of the parts
                    p.part_sent.clear()
            elif o.last is not None and e.mode != "reliable":
                # state (latest / latest_reliable, e.g. /tf_static): the new peer
                # needs the current value.  A reliable entry is a stream of events
                # (action results): one the old peer acked was applied there, and
                # re-sending it would publish it twice on the same graph.
                data, t = o.last
                self._send_new(o, data, t, now, replay=True)
        for cid, call in list(self._calls.items()):
            del self._calls[cid]
            self.srv_stats[call.service.name]["errors"] += 1
            self._defer(call.callback, None, "peer restarted")
        self._serving.clear()
        self._served.clear()
        self._defer(
            self.on_event,
            "peer_restart",
            {"robot": self.robot, "epoch": self.peer_epoch},
        )

    # ------------------------------------------------------------------ send

    def publish(self, name: str, data: bytes, t: float | None = None) -> bool:
        """A local message for outbound entry ``name`` (CDR bytes).  False when
        ``name`` is not an outbound entry of this side."""
        o = self.out.get(name)
        if o is None:
            return False
        now = self.clock()
        t = now if t is None else t
        with self._lock:
            o.last = (bytes(data), t)
            e = o.entry
            if e.mode == "latest" and e.rate_hz:
                if now - o.t_last_send < 1.0 / e.rate_hz:
                    if o.latest_wait is not None:
                        o.c["superseded"] += 1
                    o.latest_wait = (bytes(data), t)
                    return True
            self._send_new(o, bytes(data), t, now)
        self._flush()
        return True

    def _send_new(
        self, o: _Out, data: bytes, t: float, now: float, replay=False
    ) -> None:
        """Lock held: a new message for ``o`` (first transmission)."""
        e = o.entry
        tseq = o.next_tseq
        o.next_tseq += 1
        o.t_last_send = now
        o.latest_wait = None
        o.c["resent" if replay else "sent"] += 1
        if e.mode == "latest":
            self._put_data(o, tseq, data, t, None, None)
            return
        pid = self._new_id()
        if e.mode == "latest_reliable":
            o.c["superseded"] += len(o.pending)
            for old in o.pending:
                self._id_owner.pop(old, None)
            o.pending.clear()
        p = _Pending(pid, tseq, data, t, now, self.peer_epoch)
        p.parts = max(1, -(-len(data) // self.chunk_bytes))
        o.pending[pid] = p
        self._id_owner[pid] = e.name
        if e.mode == "reliable":
            while len(o.pending) > self.queue_cap:
                old, _ = o.pending.popitem(last=False)
                self._id_owner.pop(old, None)
                o.c["overflow"] += 1
        if p.parts > 1:
            self._send_parts(o, p, now)
        else:
            self._put_data(o, tseq, data, t, pid, p.to)

    def _send_parts(self, o: _Out, p: _Pending, now: float) -> None:
        """Lock held: send the parts of ``p`` not acked, keeping at most
        ``chunk_window`` of them in flight (sent less than ``retry_s`` ago)."""
        rto = max(self.retry_s, p.rto)
        in_flight = sum(
            1 for i, ts in p.part_sent.items()
            if i not in p.have and now - ts < rto
        )  # fmt: skip
        timed_out = False
        for i in range(p.parts):
            if in_flight >= self.chunk_window:
                break
            if i in p.have:
                continue
            ts = p.part_sent.get(i)
            if ts is not None and now - ts < rto:
                continue
            # a timeout with no part acked since: the link is slower (or lossier)
            # than the timer assumed
            if ts is not None and (p.last_pack_t is None or p.last_pack_t < ts):
                timed_out = True
            p.part_sent[i] = now
            p.send_n += 1
            p.part_order[i] = p.send_n
            in_flight += 1
            o.c["parts_sent"] += 1
            lo, hi = i * self.chunk_bytes, (i + 1) * self.chunk_bytes
            self._put_data(o, p.tseq, p.data[lo:hi], p.t, p.id, p.to, part=(i, p.parts))
        if timed_out:
            p.rto = min(max(rto, self.retry_s) * 2, 8.0)  # back off
            p.tries += 1  # the give-up budget counts timeouts, not ticks

    def _put_data(self, o: _Out, tseq, data, t, pid, to, part=None) -> None:
        e = o.entry
        payload = {
            "topic": e.name,
            "type": e.type,
            "th": self.type_hash(e.type),
            "mode": e.mode,
            "tseq": tseq,
            "to": to if pid is not None else self.peer_epoch,
            "data": data,
        }
        if pid is not None:
            payload["id"] = pid
        if part is not None:
            payload["part"], payload["parts"] = part
        if e.mode == "reliable" and o.pending:
            payload["lo"] = next(iter(o.pending.values())).tseq
        key = f"{self.prefix}/{self.out_dir}{e.name}"
        self._put(key, f"ros/{self.out_dir}", payload, e.cls, t)

    def call(
        self,
        name: str,
        request: bytes,
        callback: Callable[["bytes | None", "str | None"], None],
        timeout: float | None = None,
    ) -> str:
        """Relay a service request to the peer's real server.  ``callback(resp,
        err)`` exactly once: ``(bytes, None)`` or ``(None, "timeout" | error)``."""
        s = self.srv_out.get(name)
        if s is None:
            raise KeyError(f"{name!r} is not a service this side calls ({self.side})")
        now = self.clock()
        with self._lock:
            cid = self._new_id()
            payload = {
                "op": "req",
                "side": self.side,
                "id": cid,
                "service": name,
                "type": s.type,
                "th": self.type_hash(s.type),
                "to": self.peer_epoch,
                "data": bytes(request),
            }
            tmo = s.timeout_s if timeout is None else timeout
            self._calls[cid] = _Call(cid, s, payload, now, now, now + tmo, callback)
            self.srv_stats[name]["calls"] += 1
            self._put(f"{self.prefix}/srv/req", "ros/srv", payload, s.cls, now)
        self._flush()
        return cid

    # ------------------------------------------------------------------ receive

    def _ack(self, name: str, pid, ok=True, err=None, subs=None, now=0.0) -> None:
        p = {"op": "ack", "id": pid, "topic": name, "ok": ok}
        if err:
            p["err"] = err
        if subs is not None:
            p["subs"] = int(subs)
        self._put(f"{self.prefix}/ack/{self.out_dir}", "ros/ack", p, "C0", now)

    def _on_data(self, key: str, data: bytes) -> None:
        env = self._unpack(data)
        if env is None:
            return
        now = self.clock()
        deliver, acks = [], []
        with self._lock:
            if self._peer_seen(env, now):
                p = env["p"]
                name = p.get("topic")
                st = self.inn.get(name)
                if st is None:
                    self.unknown_topics += 1
                else:
                    e = self.resolved.topics[name]
                    self._receive(e, st, env, p, now, deliver, acks)
        self._flush()
        self._apply(deliver, acks)

    def _receive(
        self, e: Entry, st: _In, env, p, now, deliver: list, acks: list
    ) -> None:
        """Lock held: one inbound data message."""
        st.last_rx = now
        pid = p.get("id")
        if p.get("to") and p["to"] != self.epoch:
            st.c["stale_epoch"] += 1  # meant for a previous incarnation of this end
            if pid is not None:
                self._ack(e.name, pid, False, "stale epoch", now=now)
            return
        if st.epoch != env["epoch"]:
            st.reset(env["epoch"])
        if p.get("parts") and int(p["parts"]) > 1 and pid is not None:
            p = self._reassemble(e, st, p, pid, now)
            if p is None:
                return  # more parts to come
        tseq = int(p.get("tseq", 0))
        th, mine = p.get("th"), self.type_hash(e.type)
        err = None
        if (th and mine and th != mine) or p.get("type") not in (None, e.type):
            st.c["type_mismatch"] += 1
            err = "type mismatch"
        elif self._expired(env, e.ttl_s, now, e.ttl_strict):
            st.c["expired"] += 1
            err = "expired"
        meta = {"t": env.get("t"), "tseq": tseq, "id": pid, "epoch": env["epoch"]}
        if e.mode == "reliable":
            lo = int(p.get("lo", tseq))
            if st.expected is None:
                st.expected = min(lo, tseq)
            if tseq < st.expected or tseq in st.held:
                st.c["dup"] += 1
            else:
                # a rejected message still fills its slot, so it never blocks the queue
                st.held[tseq] = None if err else (p["data"], meta)
            self._advance(e, st, lo, deliver)
        elif err is None:
            if st.last_applied is not None and tseq <= st.last_applied:
                st.c["dup"] += 1
            else:
                st.last_applied = tseq
                deliver.append((e, p["data"], meta))
        if pid is not None:
            if err:
                self._ack(e.name, pid, False, err, now=now)
            else:
                acks.append((e.name, pid))

    def _reassemble(
        self, e: Entry, st: _In, p: dict, pid: str, now: float
    ) -> dict | None:
        """Lock held: one part of a large message.  Acks the part; returns the
        whole message (as an unsplit payload) once every part is in, else None."""
        n, i = int(p["parts"]), int(p.get("part", 0))
        st.c["parts_rx"] += 1
        self._put(f"{self.prefix}/ack/{self.out_dir}", "ros/ack",
                  {"op": "pack", "id": pid, "topic": e.name, "part": i}, "C0", now)  # fmt: skip
        if pid in st.done:
            # resent after we reassembled it: our ack was lost, say so again
            self._ack(e.name, pid, now=now)
            return None
        tseq = int(p.get("tseq", 0))
        if e.mode == "latest_reliable":
            newest = max((b["tseq"] for b in st.partial.values()), default=None)
            if (st.last_applied is not None and tseq <= st.last_applied) or (
                newest is not None and tseq < newest
            ):
                st.c["dup"] += 1
                return None  # a late part of a message a newer one replaced
            for old in [k for k, b in st.partial.items() if b["tseq"] < tseq]:
                del st.partial[old]  # superseded by a newer message
        buf = st.partial.setdefault(pid, {"n": n, "chunks": {}, "t": now, "tseq": tseq})
        buf["chunks"][i] = bytes(p["data"])
        if len(buf["chunks"]) < buf["n"]:
            return None
        del st.partial[pid]
        st.done[pid] = None
        while len(st.done) > 256:
            st.done.popitem(last=False)
        whole = {k: v for k, v in p.items() if k not in ("part", "parts")}
        whole["data"] = b"".join(buf["chunks"][k] for k in range(buf["n"]))
        return whole

    def _drop_stale_parts(self, now: float) -> None:
        """Lock held: forget reassemblies that stopped (sender gave up / restarted)."""
        for name, st in self.inn.items():
            e = self.resolved.topics.get(name)
            limit = max(60.0, (e.ttl_s or 0) * 2) if e is not None else 60.0
            for pid in [k for k, b in st.partial.items() if now - b["t"] > limit]:
                del st.partial[pid]

    @staticmethod
    def _advance(e: Entry, st: _In, lo: int, deliver: list) -> None:
        """Reliable: skip what the sender gave up on (below ``lo``), then apply in order."""
        if st.expected is None:
            return
        if lo > st.expected:
            for k in sorted(k for k in st.held if k < lo):
                item = st.held.pop(k)
                if item is not None:
                    deliver.append((e, *item))
            st.expected = lo
        while st.expected in st.held:
            item = st.held.pop(st.expected)
            if item is not None:
                deliver.append((e, *item))
            st.expected += 1

    def _apply(self, deliver: list, acks: list) -> None:
        """Outside the lock: hand messages to ``on_message`` in order, then ack
        (with the local subscriber count the handler returned)."""
        subs: dict[str, int | None] = {}
        for e, data, meta in deliver:
            n = None
            if self.on_message is not None:
                try:
                    n = self.on_message(e, data, meta)
                except Exception as exc:
                    self.log(f"relay deliver {e.name} failed: {exc}")
            subs[e.name] = n
            with self._lock:
                self.inn[e.name].c["delivered"] += 1
        if not acks:
            return
        now = self.clock()
        with self._lock:
            for name, pid in acks:
                self._ack(name, pid, subs=subs.get(name), now=now)
        self._flush()

    def _on_ack(self, key: str, data: bytes) -> None:
        env = self._unpack(data)
        if env is None:
            return
        now = self.clock()
        events, deliver = [], []
        with self._lock:
            if not self._peer_seen(env, now):
                return
            p = env["p"]
            op = p.get("op")
            if op == "hb":
                self._on_hb(env, p, now, deliver)
            elif op == "pack":
                name = self._id_owner.get(p.get("id"))
                o = self.out.get(name) if name else None
                pend = o.pending.get(p.get("id")) if o is not None else None
                if pend is not None:
                    part = int(p.get("part", -1))
                    if part not in pend.have:
                        pend.tries = 1  # progress: the retry budget restarts
                        if pend.last_pack_t is not None:
                            dt = now - pend.last_pack_t
                            pend.ack_dt = (
                                dt
                                if pend.ack_dt is None
                                else 0.7 * pend.ack_dt + 0.3 * dt
                            )
                            # a part acked every ack_dt: allow the window plus slack
                            pend.rto = min(
                                2.0 * (self.chunk_window + 1) * pend.ack_dt, 8.0
                            )
                        pend.last_pack_t = now
                    pend.have.add(part)
                    # fast retransmit: a part sent before this one and still not
                    # acked was lost (parts go out in order on one priority queue)
                    order = pend.part_order.get(part)
                    if order is not None:
                        for j, oj in pend.part_order.items():
                            if j not in pend.have and oj < order:
                                pend.part_sent[j] = -1e18
                    self._send_parts(o, pend, now)
            elif (
                op == "ack" and not p.get("ok", True) and p.get("err") == "stale epoch"
            ):
                # Sent by a new incarnation of the peer to a retry still addressed
                # to the old one.  _peer_seen() above has already handled the
                # restart: the pending was dropped, or (replay_on_resync)
                # re-addressed to the new epoch and is resent next tick.  Popping
                # it here would lose it (an action_result across a bridge restart).
                name = self._id_owner.get(p.get("id"))
                if name in self.out:
                    self.out[name].c["stale_epoch"] += 1
            elif op == "ack":
                name = self._id_owner.pop(p.get("id"), None)
                o = self.out.get(name) if name else None
                if o is not None and o.pending.pop(p.get("id"), None) is not None:
                    o.c["acked"] += 1
                    o.last_ack_t = now
                    if "subs" in p:
                        o.last_subs = p["subs"]
                    if not p.get("ok", True):
                        o.c["rejected"] += 1
                        o.last_err = p.get("err")
                    events.append(
                        {"topic": name, "ok": bool(p.get("ok", True)), "err": p.get("err"), "subs": p.get("subs")}
                    )  # fmt: skip
        self._flush()
        self._apply(deliver, [])
        for ev in events:
            self.on_event("ack", {"robot": self.robot, **ev})

    def _on_hb(self, env: dict, p: dict, now: float, deliver: list) -> None:
        """Lock held: heartbeat -> RTT, clock offset (NTP-style; the lowest-RTT
        of the last 20 samples), peer platform, config hash, and the reliable
        topics' ``lo`` (so messages held behind one the sender gave up on are
        applied even when nothing else follows)."""
        for name, lo in (p.get("lo") or {}).items():
            st = self.inn.get(name)
            e = self.resolved.topics.get(name)
            if st is not None and e.mode == "reliable" and st.epoch == env["epoch"]:
                self._advance(e, st, int(lo), deliver)
        self._peer_hb = (float(p.get("t", now)), now)
        self.peer_platform = p.get("platform") or self.peer_platform
        echo_t, held = p.get("echo_t"), p.get("held")
        if echo_t is not None and held is not None:
            rtt = max(0.0, (now - float(echo_t)) - float(held))
            self.rtt = rtt if self.rtt is None else 0.8 * self.rtt + 0.2 * rtt
            off = (float(p["t"]) - float(held)) - (float(echo_t) + rtt / 2.0)
            self._offsets.append((rtt, off))
            self.clock_offset = min(self._offsets)[1]
        cfg = p.get("cfg")
        self.peer_cfg = cfg
        if cfg and cfg != self.config.hash and not self._cfg_warned:
            self._cfg_warned = True
            msg = (
                f"relay config mismatch with {self.robot}'s peer: "
                f"mine {self.config.hash}, theirs {cfg}"
            )
            self.log(msg)
            self._defer(self.on_event, "cfg_mismatch", {"robot": self.robot, "mine": self.config.hash, "peer": cfg})  # fmt: skip

    def _on_srv_req(self, key: str, data: bytes) -> None:
        env = self._unpack(data)
        if env is None or env["p"].get("side") == self.side:
            return
        now = self.clock()
        call = None
        with self._lock:
            if not self._peer_seen(env, now):
                return
            p = env["p"]
            rid, name = p.get("id"), p.get("service")
            s = self.srv_in.get(name)
            if s is None:
                self.unknown_services += 1
                self._rep(rid, False, f"service {name!r} not relayed here", now=now)
            elif rid in self._served:
                self.srv_stats[name]["dup"] += 1
                self._put(
                    f"{self.prefix}/srv/rep", "ros/srv", self._served[rid], s.cls, now
                )
            elif rid in self._serving:
                self.srv_stats[name]["dup"] += 1
            elif p.get("to") and p["to"] != self.epoch:
                self._rep(rid, False, "stale epoch", now=now)
            elif (p.get("th") and self.type_hash(s.type) and p["th"] != self.type_hash(s.type)):  # fmt: skip
                self.srv_stats[name]["errors"] += 1
                self._rep(rid, False, "type mismatch", now=now)
            elif self._expired(env, s.ttl_s, now):
                self.srv_stats[name]["errors"] += 1
                self._rep(rid, False, "expired", now=now)
            elif self.on_service is None:
                self._rep(rid, False, "no service handler", now=now)
            else:
                self._serving.add(rid)
                self.srv_stats[name]["served"] += 1
                call = (s, bytes(p.get("data") or b""), rid)
        self._flush()
        if call is not None:
            s, req, rid = call

            def reply(resp, err=None, rid=rid, s=s):
                with self._lock:
                    if rid not in self._serving:
                        return  # peer restarted meanwhile
                    self._serving.discard(rid)
                    self._rep(
                        rid, resp is not None and not err, err, resp, s, self.clock()
                    )
                self._flush()

            try:
                self.on_service(s, req, reply)
            except Exception as exc:
                reply(None, f"{type(exc).__name__}: {exc}")

    def _rep(
        self, rid, ok, err=None, data=None, s: ServiceEntry | None = None, now=0.0
    ):
        """Lock held: queue (and cache) a service reply."""
        p = {"op": "rep", "side": self.side, "id": rid, "ok": bool(ok)}
        if err:
            p["err"] = err
        if data is not None:
            p["data"] = bytes(data)
        if s is not None:
            self._served[rid] = p
            while len(self._served) > self.srv_cache:
                self._served.popitem(last=False)
        self._put(f"{self.prefix}/srv/rep", "ros/srv", p, s.cls if s else "C0", now)

    def _on_srv_rep(self, key: str, data: bytes) -> None:
        env = self._unpack(data)
        if env is None or env["p"].get("side") == self.side:
            return
        now = self.clock()
        with self._lock:
            if not self._peer_seen(env, now):
                return
            p = env["p"]
            call = self._calls.pop(p.get("id"), None)
            if call is not None:
                st = self.srv_stats[call.service.name]
                if p.get("ok"):
                    st["ok"] += 1
                    self._defer(call.callback, p.get("data") or b"", None)
                else:
                    st["errors"] += 1
                    self._defer(call.callback, None, p.get("err") or "error")
        self._flush()

    # ------------------------------------------------------------------ tick

    def tick(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        with self._lock:
            for o in self.out.values():
                self._tick_out(o, now)
            self._drop_stale_parts(now)
            for cid, call in list(self._calls.items()):
                if now >= call.deadline:
                    del self._calls[cid]
                    self.srv_stats[call.service.name]["timeouts"] += 1
                    self._defer(call.callback, None, "timeout")
                elif now - call.t_sent >= self.retry_s:
                    call.t_sent = now
                    call.payload["to"] = self.peer_epoch
                    self._put(f"{self.prefix}/srv/req", "ros/srv", call.payload, call.service.cls, call.t)  # fmt: skip
            if now - self._t_hb >= self.hb_period:
                self._t_hb = now
                hb = {
                    "op": "hb",
                    "t": now,
                    "cfg": self.config.hash,
                    "platform": self.platform,
                    "side": self.side,
                    "echo_t": None,
                    "held": None,
                }
                if self._peer_hb is not None:
                    hb["echo_t"], hb["held"] = self._peer_hb[0], now - self._peer_hb[1]
                lo = {
                    n: (
                        next(iter(o.pending.values())).tseq
                        if o.pending
                        else o.next_tseq
                    )
                    for n, o in self.out.items()
                    if o.entry.mode == "reliable" and o.next_tseq
                }
                if lo:
                    hb["lo"] = lo
                self._put(f"{self.prefix}/ack/{self.out_dir}", "ros/ack", hb, "C0", now)
            if self._loss_t is None:
                self._loss_t = now
            elif now - self._loss_t >= self.loss_window_s:
                got, lost = self._loss_win
                self.loss = lost / (got + lost) if got + lost else 0.0
                self._loss_win = [0, 0]
                self._loss_t = now
        self._flush()

    def _tick_out(self, o: _Out, now: float) -> None:
        e = o.entry
        if o.latest_wait is not None and (
            not e.rate_hz or now - o.t_last_send >= 1.0 / e.rate_hz
        ):
            data, t = o.latest_wait
            if e.ttl_s is not None and now - t > e.ttl_s:
                o.latest_wait = None  # went stale waiting for its slot
                o.c["expired"] += 1
            else:
                self._send_new(o, data, t, now)
        for pid, p in list(o.pending.items()):
            if e.ttl_s is not None and now - p.t > e.ttl_s:
                del o.pending[pid]
                self._id_owner.pop(pid, None)
                o.c["expired"] += 1
                continue
            if p.parts > 1:
                if p.tries >= self.max_tries:
                    del o.pending[pid]
                    self._id_owner.pop(pid, None)
                    o.c["gave_up"] += 1
                    continue
                # resends the parts not acked within the part timer; when every part
                # is acked but the final ack was lost, resends the last part so the
                # receiver acks the whole message again
                if len(p.have) == p.parts and now - p.t_sent >= max(
                    self.retry_s, p.rto
                ):
                    p.t_sent = now
                    p.have.discard(p.parts - 1)
                    p.part_sent.pop(p.parts - 1, None)
                self._send_parts(o, p, now)
                continue
            if now - p.t_sent < self.retry_s:
                continue
            if p.tries >= self.max_tries:
                del o.pending[pid]
                self._id_owner.pop(pid, None)
                o.c["gave_up"] += 1
                continue
            p.tries += 1
            p.t_sent = now
            if p.to is None:
                p.to = self.peer_epoch
            o.c["resent"] += 1
            self._put_data(o, p.tseq, p.data, p.t, pid, p.to)

    # ------------------------------------------------------------------ status

    def connected(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        return (
            self.last_seen is not None and now - self.last_seen < self.connected_timeout
        )

    def acked(self, name: str) -> int:
        o = self.out.get(name)
        return 0 if o is None else o.c["acked"]

    def status(self, now: float | None = None) -> dict:
        """JSON-able link + per-entry state (what ``fleetlink/link_status`` carries)."""
        now = self.clock() if now is None else now
        with self._lock:
            topics = {}
            for name, o in self.out.items():
                topics[name] = {
                    "dir": o.entry.dir,
                    "mode": o.entry.mode,
                    "pending": len(o.pending),
                    "last_ack_t": o.last_ack_t,
                    "subs": o.last_subs,
                    "err": o.last_err,
                    **o.c,
                }
            for name, st in self.inn.items():
                e = self.resolved.topics[name]
                topics[name] = {
                    "dir": e.dir,
                    "mode": e.mode,
                    "held": len(st.held),
                    "last_rx_t": st.last_rx,
                    **st.c,
                }
            return {
                "robot": self.robot,
                "side": self.side,
                "connected": self.connected(now),
                "rtt": self.rtt,
                "loss": round(self.loss, 4),
                "last_seen": self.last_seen,
                "age_s": None if self.last_seen is None else now - self.last_seen,
                "clock_offset": self.clock_offset,
                "epoch": self.epoch,
                "peer_epoch": self.peer_epoch,
                "peer_restarts": self.peer_restarts,
                "platform": self.platform or self.peer_platform,
                "cfg_hash": self.config.hash,
                "peer_cfg_hash": self.peer_cfg,
                "cfg_match": None
                if self.peer_cfg is None
                else self.peer_cfg == self.config.hash,
                "topics": topics,
                "services": {
                    n: {"dir": s.dir, **self.srv_stats[n]}
                    for n, s in self.resolved.services.items()
                },  # fmt: skip
                "unknown_topics": self.unknown_topics,
            }

    def close(self) -> None:
        unsub = getattr(self.transport, "unsubscribe", None)
        if unsub is not None:
            for s in self._subs:
                try:
                    unsub(s)
                except Exception:
                    pass
        else:
            for s in self._subs:
                try:
                    s.undeclare()
                except Exception:
                    pass
        self._subs = []
        with self._lock:
            calls, self._calls = list(self._calls.values()), {}
        for c in calls:
            try:
                c.callback(None, "closed")
            except Exception:
                pass
