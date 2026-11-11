"""Health reports for ``ros_system_monitor`` (``NodeInfoMsg`` on ``~/node_status``).

The agent and the bridge publish one ``ros_system_monitor_msgs/NodeInfoMsg`` per
``status_period_s`` (1 Hz) with the nicknames ``fleetlink_agent`` /
``fleetlink_bridge`` (param ``status_nickname``).  The status is computed here
from what the link already tracks (the relay heartbeat's liveness and RTT, the
console's ``ping`` / ``fb`` requests, the zenoh session, the mesh mode), so this
module needs no ROS and adds nothing to the wire.  ``ros_system_monitor_msgs`` is
optional: without it :class:`StatusReporter` logs once and publishes nothing.

Levels are the ``NodeInfoMsg`` constant names: ``NOMINAL``, ``WARNING``, ``ERROR``.
"""

from __future__ import annotations

from typing import Callable

NOMINAL = "NOMINAL"
WARNING = "WARNING"
ERROR = "ERROR"
_RANK = {NOMINAL: 0, WARNING: 1, ERROR: 2}

# relay RTT (EWMA of the heartbeat round trip) above which the agent warns
RTT_WARN_S = 1.5
BRIDGE_IDLE_NOTES = "Idle: router mode, robot topics ride rmw_zenoh"


def _worst(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


def fmt_s(s: float | None) -> str:
    """Age / duration: ``850ms``, ``4.2s``, ``3m``."""
    if s is None:
        return "?"
    s = max(0.0, float(s))
    if s < 1.0:
        return f"{s * 1000:.0f}ms"
    if s < 60.0:
        return f"{s:.1f}s"
    if s < 3600.0:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"


def fmt_bytes(n: float) -> str:
    n = float(n)
    if n < 1e3:
        return f"{n:.0f}B"
    if n < 1e6:
        return f"{n / 1e3:.1f}kB"
    return f"{n / 1e6:.1f}MB"


def fmt_endpoints(endpoints: list[str] | None) -> str:
    eps = list(endpoints or [])
    if not eps:
        return "zenoh ?"
    return eps[0] + (f" (+{len(eps) - 1} failover)" if len(eps) > 1 else "")


def agent_status(
    *,
    opened: bool,
    open_error: str | None = None,
    endpoints: list[str] | None = None,
    relay: dict | None = None,
    console_age_s: float | None = None,
    console_rtt_s: float | None = None,
    mesh_source: str | None = None,
    mesh_fallback: str | None = None,
    queued_bytes: float | None = None,
    store_error: str | None = None,
    rtt_warn_s: float = RTT_WARN_S,
) -> tuple[str, str]:
    """``(level, notes)`` for the robot's ``fleetlink_agent``.

    ``relay`` is ``RelayCore.status()`` when the ROS relay runs (fleetlink mode)
    and None otherwise (router mode).  In fleetlink mode the base link is the
    relay heartbeat: down (or never seen) or an RTT over ``rtt_warn_s`` warns.
    In router mode nothing on the link says whether a base should be there, so
    the console's last request is a note only.
    """
    level = NOMINAL
    parts: list[str] = []
    if not opened:
        err = open_error or "opening"
        return ERROR, f"zenoh session not open: {err} | {fmt_endpoints(endpoints)}"
    if relay is not None:
        rtt = relay.get("rtt")
        if relay.get("connected"):
            link = "base up"
            if rtt is not None:
                link += f" rtt {fmt_s(rtt)}"
                if rtt > rtt_warn_s:
                    level = WARNING
                    link += " (high)"
            loss = relay.get("loss") or 0.0
            if loss >= 0.01:
                link += f" loss {loss * 100:.0f}%"
        elif relay.get("last_seen") is None:
            level = WARNING
            link = "no base heartbeat yet"
        else:
            level = WARNING
            link = f"base link down {fmt_s(relay.get('age_s'))}"
        if relay.get("cfg_match") is False:
            level = WARNING
            link += " (relay.yaml differs from base)"
        parts += ["fleetlink", link]
    else:
        if console_age_s is None:
            console = "no console yet"
        else:
            console = f"console {fmt_s(console_age_s)} ago"
            if console_rtt_s is not None:
                console += f" rtt {fmt_s(console_rtt_s)}"
        parts += ["router", console]
    parts.append(fmt_endpoints(endpoints))
    if mesh_fallback:
        level = _worst(level, WARNING)
        parts.append(f"mesh dsg (fallback: {mesh_fallback})")
    elif mesh_source:
        parts.append(f"mesh {mesh_source}")
    if queued_bytes:
        parts.append(f"queued {fmt_bytes(queued_bytes)}")
    if store_error:
        parts.append(f"store mirror off: {store_error}")
    return level, " | ".join(parts)


def bridge_status(
    robots: dict[str, dict],
    *,
    opened: bool = True,
    open_error: str | None = None,
    idle: bool = False,
) -> tuple[str, str]:
    """``(level, notes)`` for the base's ``fleetlink_bridge``.

    ``robots`` maps robot -> ``RelayCore.status()``.  A robot never heard from is
    listed as waiting (it may simply not be running); one that was heard and went
    silent warns, as does a relay.yaml that differs between the two ends.
    """
    if idle:
        return NOMINAL, BRIDGE_IDLE_NOTES
    if not opened:
        return ERROR, f"zenoh session not open: {open_error or 'opening'}"
    level = NOMINAL
    up, down, waiting = [], [], []
    for name in sorted(robots):
        st = robots[name] or {}
        if st.get("last_seen") is None:
            waiting.append(name)
            continue
        if st.get("connected"):
            rtt = st.get("rtt")
            up.append(name + (f" {fmt_s(rtt)}" if rtt is not None else ""))
        else:
            level = WARNING
            down.append(f"{name} silent {fmt_s(st.get('age_s'))}")
        if st.get("cfg_match") is False:
            level = WARNING
            down.append(f"{name} relay.yaml differs")
    parts = []
    if up:
        parts.append("up: " + ", ".join(up))
    if down:
        parts.append("DOWN: " + ", ".join(down))
    if waiting:
        parts.append("waiting: " + ", ".join(waiting))
    return level, " | ".join(parts) or "no robots yet"


class StatusReporter:
    """Publishes ``compute() -> (level, notes)`` as a ``NodeInfoMsg`` on
    ``~/node_status`` every ``period_s`` from an rclpy timer.  A no-op (one log
    line) when ``ros_system_monitor_msgs`` is not installed or ``period_s <= 0``."""

    def __init__(
        self,
        node,
        compute: Callable[[], tuple[str, str]],
        nickname: str,
        period_s: float = 1.0,
    ):
        self.node = node
        self.compute = compute
        self.nickname = nickname
        self.pub = None
        log = node.get_logger()
        if period_s <= 0:
            return
        try:
            from ros_system_monitor_msgs.msg import NodeInfoMsg
        except ImportError:
            log.info("ros_system_monitor_msgs not installed: no ~/node_status")
            return
        self._msg = NodeInfoMsg
        self.pub = node.create_publisher(NodeInfoMsg, "~/node_status", 1)
        self.timer = node.create_timer(period_s, self.publish)

    def publish(self) -> None:
        try:
            level, notes = self.compute()
        except Exception as exc:  # noqa: BLE001  never take the node down
            level, notes = ERROR, f"status failed: {type(exc).__name__}: {exc}"
        msg = self._msg()
        msg.nickname = self.nickname
        msg.node_name = self.node.get_fully_qualified_name()
        msg.status = getattr(self._msg, level)
        msg.notes = notes
        self.pub.publish(msg)
