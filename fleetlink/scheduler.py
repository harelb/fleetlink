"""Per-class token-bucket scheduler drained in strict priority C0 > C1 > C2 > C3.

Items are queued with a ``build`` callable that produces the wire bytes at send
time, so sequence numbers and envelope timestamps are assigned when a message
actually leaves (dropping or re-prioritising queued items never creates seq
gaps).  Buckets allow a burst of ``burst_s`` seconds of budget and may go into
debt by one message, so long-run throughput equals the budget.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

from fleetlink.transport import CLASSES, Transport

AUTO_CLASSES = ("C2", "C3")


@dataclass
class Item:
    cls: str
    key: str
    build: Callable[[], bytes]
    size_hint: int = 0
    on_sent: Callable[[int], None] | None = None
    tag: str | None = None  # items with the same (cls, tag) replace each other


class Scheduler:
    def __init__(
        self,
        transport: Transport,
        budgets: dict,
        clock: Callable[[], float],
        burst_s: float = 1.0,
        total_bps: float | None = None,
        budget_min: dict | None = None,
        budget_max: dict | None = None,
        auto: bool = False,
        loss_threshold: float = 0.05,
        ai_step_frac: float = 0.1,
        md_factor: float = 0.7,
        md_interval_s: float = 2.0,
    ):
        self.transport = transport
        self.clock = clock
        self.burst_s = burst_s
        self.budgets = {c: budgets.get(c) for c in CLASSES}  # None = unlimited
        self.total_bps = total_bps
        self.budget_min = {"C2": 5_000, "C3": 5_000, **(budget_min or {})}
        self.budget_max = {"C2": 500_000, "C3": 1_000_000, **(budget_max or {})}
        self.auto = auto
        self.loss_threshold = loss_threshold
        self.ai_step_frac = ai_step_frac
        self.md_factor = md_factor
        self.md_interval_s = md_interval_s
        self.decreases = 0
        self.queues: dict[str, deque[Item]] = {c: deque() for c in CLASSES}
        self.tokens = {c: self._cap(c) for c in CLASSES}
        self.total_tokens = self._total_cap()
        self.sent_bytes = dict.fromkeys(CLASSES, 0)
        self.sent_msgs = dict.fromkeys(CLASSES, 0)
        self.rate = dict.fromkeys(CLASSES, 0.0)  # EWMA bytes/s actually put
        self._win = dict.fromkeys(CLASSES, 0)
        self._win_t: float | None = None
        self._last: float | None = None
        self._last_decrease = -1e9
        self._lock = threading.RLock()

    # -- budgets -------------------------------------------------------------

    def _cap(self, cls: str) -> float:
        b = self.budgets[cls]
        return float("inf") if b is None else max(1.0, b * self.burst_s)

    def _total_cap(self) -> float:
        return float("inf") if not self.total_bps else self.total_bps * self.burst_s

    def set_budget(self, cls: str, bps: float | None) -> None:
        if cls not in CLASSES:
            raise ValueError(f"unknown class {cls!r}")
        with self._lock:
            self.budgets[cls] = None if bps is None else max(0.0, float(bps))
            self.tokens[cls] = min(self.tokens[cls], self._cap(cls))

    def budget(self, cls: str) -> float | None:
        return self.budgets[cls]

    # -- queue ---------------------------------------------------------------

    def enqueue(self, item: Item, front: bool = False) -> None:
        with self._lock:
            q = self.queues[item.cls]
            if item.tag is not None:
                for i, other in enumerate(q):
                    if other.tag == item.tag:
                        if front:
                            del q[i]
                            break
                        q[i] = item
                        return
            if front:
                q.appendleft(item)
            else:
                q.append(item)

    def clear(self, cls: str, predicate: Callable[[Item], bool] | None = None) -> int:
        with self._lock:
            q = self.queues[cls]
            if predicate is None:
                n = len(q)
                q.clear()
                return n
            keep = [it for it in q if not predicate(it)]
            n = len(q) - len(keep)
            self.queues[cls] = deque(keep)
            return n

    def queued_bytes(self, cls: str) -> int:
        with self._lock:
            return sum(it.size_hint for it in self.queues[cls])

    def queue_stats(self) -> dict:
        with self._lock:
            return {
                c: {"n": len(q), "bytes": sum(it.size_hint for it in q)}
                for c, q in self.queues.items()
            }

    # -- draining ------------------------------------------------------------

    def _refill(self, now: float) -> None:
        if self._last is None:
            self._last = now
            self._win_t = now
            return
        dt = max(0.0, now - self._last)
        self._last = now
        for c in CLASSES:
            b = self.budgets[c]
            if b is not None:
                self.tokens[c] = min(self._cap(c), self.tokens[c] + b * dt)
        if self.total_bps:
            self.total_tokens = min(
                self._total_cap(), self.total_tokens + self.total_bps * dt
            )
        wdt = now - self._win_t
        if wdt >= 1.0:
            for c in CLASSES:
                r = self._win[c] / wdt
                self.rate[c] = 0.5 * self.rate[c] + 0.5 * r
                self._win[c] = 0
            self._win_t = now

    def drain(self, now: float | None = None) -> dict:
        """Send everything the buckets allow, strict priority. Returns bytes sent per class."""
        now = self.clock() if now is None else now
        sent = dict.fromkeys(CLASSES, 0)
        with self._lock:
            self._refill(now)
            for c in CLASSES:
                q = self.queues[c]
                while q and self.tokens[c] > 0 and self.total_tokens > 0:
                    if not self.transport.can_send(c):
                        break
                    item = q.popleft()
                    data = item.build()
                    if data is None:  # builder decided there is nothing to send
                        continue
                    n = len(data)
                    self.transport.put(item.key, data, c)
                    if self.budgets[c] is not None:
                        self.tokens[c] -= n
                    if self.total_bps:
                        self.total_tokens -= n
                    self.sent_bytes[c] += n
                    self.sent_msgs[c] += 1
                    self._win[c] += n
                    sent[c] += n
                    if item.on_sent is not None:
                        item.on_sent(n)
        return sent

    # -- auto (AIMD) ---------------------------------------------------------

    def feedback(
        self,
        rx: dict,
        rtt: float | None = None,
        now: float | None = None,
        base_rtt: float | None = None,
        loss: dict | None = None,
    ) -> dict:
        """Adjust C2/C3 budgets from the receiver's feedback (C0 ``fb``, or the
        ``rx`` of a legacy ``ping``): ``rx`` = goodput in bytes/s per class,
        ``loss`` = fraction of messages lost per class (seq gaps), ``rtt`` in s.

        Multiplicative decrease (x``md_factor``, at most every ``md_interval_s``)
        when a class's loss exceeds ``loss_threshold`` (all classes are zenoh
        DROP, so a congested link shows up as seq gaps), when its goodput falls
        below 80% of what we put, or when the RTT inflates (queueing).
        Additive increase (+``ai_step_frac`` of max) when the class is
        backlogged and the link keeps up.  Budgets stay within
        [budget_min, budget_max].  No-op unless ``auto``.
        """
        now = self.clock() if now is None else now
        with self._lock:
            if not self.auto:
                return dict(self.budgets)
            congested_rtt = (
                rtt is not None and base_rtt is not None and rtt > 2.0 * base_rtt + 0.25
            )
            may_decrease = now - self._last_decrease >= self.md_interval_s
            decreased = False
            for c in AUTO_CLASSES:
                b = self.budgets[c] or self.budget_min[c]
                tx = self.rate[c]
                got = float(rx.get(c, 0.0))
                lost = float((loss or {}).get(c) or 0.0)
                lossy = lost > self.loss_threshold or (tx > 1000 and got < 0.8 * tx)
                if lossy or congested_rtt:
                    if may_decrease:
                        b *= self.md_factor
                        decreased = True
                elif self.queues[c] and got >= 0.9 * tx:
                    b += self.ai_step_frac * self.budget_max[c]
                self.budgets[c] = min(self.budget_max[c], max(self.budget_min[c], b))
                self.tokens[c] = min(self.tokens[c], self._cap(c))
            if decreased:
                self._last_decrease = now
                self.decreases += 1
            return dict(self.budgets)
