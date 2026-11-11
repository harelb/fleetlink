"""Multi-router fleet harness: the real fleetlink-mode topology as separate processes.

::

    robot r1 graph (domain D1)                         base graph (domain DB)
    ┌──────────────────────────────┐                  ┌──────────────────────────────┐
    │ rmw_zenohd :r1_ros           │                  │ rmw_zenohd :base_ros         │
    │  ├ fleetlink-agent (relay) ──┼─▶ FaultProxy ─┐  │  ├ fleetlink-bridge ──┐      │
    │  └ node_host --role robot    │   :r1_proxy   │  │  └ node_host --role base    │
    └──────────────────────────────┘               ▼  └───────────────────────┼──────┘
    robot r2 graph (domain D2), same shape ──▶ fleetlink-router :fleetlink ◀──┘
                                                (bin/fleetlink-router, json5)

Every box is its own process (``start_new_session``; teardown is ``killpg``),
every ROS process points its rmw_zenoh session at its own router with
``ZENOH_CONFIG_OVERRIDE`` and its own ``ROS_DOMAIN_ID``.  Each robot's agent
reaches the fleetlink router only through a :class:`FaultProxy`, which can stall,
throttle or reset that robot's radio link.  An in-process zenoh observer on the
fleetlink router (what the console receiver sees) records C1 ``state`` arrivals.

Throwaway ports only (``FLEETLINK_HARNESS_PORT``, default 17470..17477); never
7447.  Every spawned process carries ``FLEETLINK_HARNESS_TAG=<tag>`` in its
environment so :func:`stray_pids` can prove nothing is left behind.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
BIN = os.path.join(PKG, "bin")
NODE_HOST = os.path.join(HERE, "harness_nodes", "node_host.py")
RMW_ZENOHD = "/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd"
TAG_VAR = "FLEETLINK_HARNESS_TAG"
LIVE_PORT = 7447


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def wait(pred, timeout: float, step: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return bool(pred())


def stray_pids(tag: str) -> list[int]:
    """PIDs of live processes whose environment carries ``TAG_VAR=tag``."""
    needle = f"{TAG_VAR}={tag}".encode()
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(f"/proc/{d}/environ", "rb") as f:
                env = f.read().split(b"\0")
            with open(f"/proc/{d}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        if needle in env and state != "Z":
            out.append(int(d))
    return out


# ----------------------------------------------------------------------------
# processes
# ----------------------------------------------------------------------------


class Proc:
    """One spawned process (own session/process group), output to a log file."""

    def __init__(self, name: str, cmd: list[str], env: dict, log_dir: str, stdin=False):
        self.name = name
        self.cmd = cmd
        self.env = env
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self.stdin = stdin
        self.proc: subprocess.Popen | None = None
        self.starts = 0

    def start(self) -> "Proc":
        self.starts += 1
        log = open(self.log_path, "ab")
        log.write(f"\n=== start #{self.starts} {time.strftime('%T')}: {self.cmd}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(
            self.cmd,
            env=self.env,
            stdin=subprocess.PIPE if self.stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE if self.stdin else log,
            stderr=log,
            start_new_session=True,
            text=self.stdin,
            bufsize=1 if self.stdin else -1,
        )
        log.close()
        return self

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, sig=signal.SIGINT, timeout: float = 5.0) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        for s, t in ((sig, timeout), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(p.pid, s)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=t)
                break
            except subprocess.TimeoutExpired:
                continue
        # the group may outlive its leader (e.g. ros2 run's child)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for f in (p.stdin, p.stdout):
            try:
                f and f.close()
            except Exception:
                pass

    def log_text(self) -> str:
        try:
            with open(self.log_path, errors="replace") as f:
                return f.read()
        except OSError:
            return ""


class Helper(Proc):
    """A ``node_host.py`` process: JSON-line commands in, JSON-line events out."""

    def __init__(self, *a, **kw):
        super().__init__(*a, stdin=True, **kw)
        self.events: list[dict] = []
        self._cv = threading.Condition()

    def start(self) -> "Helper":
        super().start()
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        return self

    def _read(self, proc) -> None:
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            ev.setdefault("rx", time.time())
            ev["rx"] = time.time()
            with self._cv:
                self.events.append(ev)
                self._cv.notify_all()

    def send(self, **cmd) -> None:
        if not self.alive():
            raise RuntimeError(f"{self.name} is not running")
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()

    def mark(self) -> int:
        with self._cv:
            return len(self.events)

    def find(self, pred, since: int = 0) -> list[dict]:
        with self._cv:
            return [e for e in self.events[since:] if pred(e)]

    def wait_for(self, pred, timeout: float, since: int = 0) -> dict | None:
        deadline = time.time() + timeout
        with self._cv:
            while True:
                for e in self.events[since:]:
                    if pred(e):
                        return e
                left = deadline - time.time()
                if left <= 0:
                    return None
                self._cv.wait(min(left, 0.2))

    def ask(self, cmd: str, ev: str, timeout: float = 10.0, **kw) -> dict | None:
        m = self.mark()
        self.send(cmd=cmd, **kw)
        return self.wait_for(lambda e: e.get("ev") == ev, timeout, since=m)

    def stop(self, sig=signal.SIGINT, timeout: float = 5.0) -> None:
        if self.alive():
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                pass
        super().stop(sig, timeout)


# ----------------------------------------------------------------------------
# FaultProxy
# ----------------------------------------------------------------------------


class _Bucket:
    def __init__(self):
        self.tokens = 0.0
        self.t = time.monotonic()
        self.lock = threading.Lock()


class FaultProxy:
    """A TCP proxy for one robot's radio link: ``127.0.0.1:listen -> target``.

    * :meth:`stall` stops forwarding in both directions but keeps every socket
      open (bytes pile up in kernel buffers, the sender's TCP stalls: a radio
      fade, not a disconnect).  New connections are accepted and held without
      an upstream until :meth:`resume`.
    * :meth:`throttle` caps each direction to ``bps`` bytes/s (token bucket
      shared by all connections), with small socket buffers so the backlog sits
      in the sender's queue, where zenoh's priorities can act, like a radio.
    * :meth:`close` resets every connection (RST).  :meth:`resume` clears
      stall and throttle.

    ``accepted`` counts downstream connections: a zenoh transport that closed
    and reopened shows up as a new one.
    """

    def __init__(self, listen_port: int, target_port: int, host: str = "127.0.0.1",
                 sockbuf: int = 4096, name: str = "proxy"):  # fmt: skip
        assert LIVE_PORT not in (listen_port, target_port)
        self.name = name
        self.host = host
        self.listen_port = listen_port
        self.target_port = target_port
        self.sockbuf = sockbuf
        self.endpoint = f"tcp/{host}:{listen_port}"
        self.stalled = False
        self.rate: float | None = None
        self.accepted = 0
        self.upstream_failures = 0
        self.bytes = defaultdict(int)  # "up" (robot -> base) / "down"
        self.events: list[tuple[float, str]] = []
        self._cv = threading.Condition()
        self._conns: list[tuple[socket.socket, socket.socket | None]] = []
        self._buckets = {"up": _Bucket(), "down": _Bucket()}
        self._running = False
        self._lsock: socket.socket | None = None

    # -- control ----------------------------------------------------------------

    def start(self) -> "FaultProxy":
        ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.sockbuf)
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sockbuf)
        ls.bind((self.host, self.listen_port))
        ls.listen(16)
        ls.settimeout(0.2)
        self._lsock = ls
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True, name=f"{self.name}-accept").start()
        return self

    def _event(self, what: str) -> None:
        self.events.append((time.time(), what))

    def stall(self) -> None:
        with self._cv:
            self.stalled = True
            self._event("stall")
            self._cv.notify_all()

    def throttle(self, bps: float | None) -> None:
        with self._cv:
            self.rate = None if not bps else float(bps)
            for b in self._buckets.values():
                with b.lock:
                    b.tokens, b.t = 0.0, time.monotonic()
            self._event(f"throttle {bps}")
            self._cv.notify_all()

    def resume(self) -> None:
        with self._cv:
            self.stalled = False
            self.rate = None
            self._event("resume")
            self._cv.notify_all()

    def close(self) -> None:
        """Reset (RST) every open connection."""
        with self._cv:
            conns, self._conns = self._conns, []
            self._event("close")
            self._cv.notify_all()
        for pair in conns:
            for s in pair:
                if s is not None:
                    _reset(s)

    def stop(self) -> None:
        self._running = False
        if self._lsock is not None:
            try:
                self._lsock.close()
            except OSError:
                pass
        self.resume()
        self.close()

    # -- forwarding -------------------------------------------------------------

    def _accept_loop(self) -> None:
        while self._running:
            try:
                cs, _ = self._lsock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            cs.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._cv:
                self.accepted += 1
                self._event("accept")
            threading.Thread(target=self._connect, args=(cs,), daemon=True).start()

    def _flowing(self, sock_alive) -> bool:
        """Block while stalled; False when the proxy is stopping."""
        with self._cv:
            while self.stalled and self._running and sock_alive():
                self._cv.wait(0.2)
            return self._running

    def _connect(self, cs: socket.socket) -> None:
        pair = [cs, None]
        with self._cv:
            self._conns.append(pair)
        if not self._flowing(lambda: pair in self._conns):
            _reset(cs)
            return
        us = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        us.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.sockbuf)
        us.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sockbuf)
        us.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        us.settimeout(3.0)
        try:
            us.connect((self.host, self.target_port))
        except OSError:
            with self._cv:
                self.upstream_failures += 1
                if pair in self._conns:
                    self._conns.remove(pair)
            us.close()
            _reset(cs)
            return
        us.settimeout(None)
        pair[1] = us
        for src, dst, d in ((cs, us, "up"), (us, cs, "down")):
            threading.Thread(target=self._pump, args=(pair, src, dst, d), daemon=True).start()

    def _take(self, direction: str, n: int) -> None:
        while True:
            rate = self.rate
            if rate is None:
                return
            b = self._buckets[direction]
            with b.lock:
                now = time.monotonic()
                b.tokens = min(rate * 0.1, b.tokens + (now - b.t) * rate)  # 100 ms burst
                b.t = now
                if b.tokens >= n or (b.tokens >= 0 and n > rate * 0.1):
                    b.tokens -= n
                    return
                need = (n - b.tokens) / rate
            time.sleep(min(need, 0.05))

    def _pump(self, pair, src: socket.socket, dst: socket.socket, direction: str) -> None:
        alive = lambda: pair in self._conns  # noqa: E731
        try:
            while self._running and alive():
                if not self._flowing(alive):
                    break
                rate = self.rate
                chunk = 4096 if rate is None else max(256, min(4096, int(rate / 20)))
                data = src.recv(chunk)
                if not data:
                    break
                # a fade that started while we were blocked in recv holds this too
                if not self._flowing(alive):
                    break
                self._take(direction, len(data))
                dst.sendall(data)
                self.bytes[direction] += len(data)
        except OSError:
            pass
        with self._cv:
            if pair in self._conns:
                self._conns.remove(pair)
        for s in pair:
            if s is not None:
                _reset(s)


def _reset(s: socket.socket) -> None:
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    except OSError:
        pass
    try:
        s.shutdown(socket.SHUT_RD)
    except OSError:
        pass
    try:
        s.close()
    except OSError:
        pass


# ----------------------------------------------------------------------------
# Fleet
# ----------------------------------------------------------------------------


TOPOLOGY = """\
# written by fleetlink/tests/harness.py
zenoh:
  port: {ros_port}
  fleetlink_port: {fl_port}
fleet_defaults:
  fleet_link: fleetlink
  fleetlink_base: {base}
machines:
  {base}:
    role: base_station
    addresses: {{lo: 127.0.0.1}}
{robots}
"""


class Fleet:
    """Base + fleetlink router + N robots, each its own set of processes.

    ``fleet.robots[r]`` has ``proxy`` (:class:`FaultProxy`), ``agent`` and
    ``helper`` (:class:`Helper`, ``--role robot``), ``router`` (its ROS router).
    ``fleet.base`` is the base helper, ``fleet.bridge`` the fleetlink-bridge,
    ``fleet.flrouter`` the fleetlink router.  ``fleet.state_times[r]`` are the
    arrival times of r's C1 ``state`` at the fleetlink router.
    """

    def __init__(self, tmp: str, robots=("r1", "r2"), base: str = "willow",
                 port: int | None = None, domain: int | None = None,
                 exec_delay: float = 0.5):  # fmt: skip
        port = int(port or os.environ.get("FLEETLINK_HARNESS_PORT", 17470))
        domain = int(domain or os.environ.get("FLEETLINK_HARNESS_DOMAIN", 94))
        n = len(robots)
        self.tmp = tmp
        self.base_name = base
        self.robot_names = list(robots)
        self.tag = secrets.token_hex(6)
        self.ports = {"base_ros": port, "fleetlink": port + 1}
        self.domains = {"base": domain}
        for i, r in enumerate(robots):
            self.ports[f"{r}_ros"] = port + 2 + i
            self.ports[f"{r}_proxy"] = port + 2 + n + i
            self.domains[r] = domain + 1 + i
        assert LIVE_PORT not in self.ports.values()
        self.exec_delay = exec_delay
        self.procs: list[Proc] = []
        self.proxies: list[FaultProxy] = []
        self.robots: dict[str, dict] = {r: {} for r in robots}
        self.state_times: dict[str, list[float]] = defaultdict(list)
        self.bulk_bytes: dict[str, int] = defaultdict(int)
        self.grid_rx: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._obs = None
        self.topology = os.path.join(tmp, "topology.yaml")

    # -- environment -------------------------------------------------------------

    def _env(self, ros_port: int | None = None, domain: int | None = None) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in ("ZENOH_CONFIG_OVERRIDE", "ZENOH_SESSION_CONFIG_URI", "ZENOH_ROUTER_CONFIG_URI")}  # fmt: skip
        env.update({
            TAG_VAR: self.tag,
            "RMW_IMPLEMENTATION": "rmw_zenoh_cpp",
            "PYTHONUNBUFFERED": "1",
            "_FLEETLINK_REEXEC": "1",  # we already run spark_env's python
            "RUST_LOG": os.environ.get("FLEETLINK_HARNESS_RUST_LOG", "warn"),
            "ADT4_ROBOT_NAME": self.base_name,
        })  # fmt: skip
        env.pop("ROS_LOCALHOST_ONLY", None)
        if ros_port is not None:
            env["ZENOH_CONFIG_OVERRIDE"] = (
                f'connect/endpoints=["tcp/127.0.0.1:{ros_port}"];'
                "scouting/multicast/enabled=false"
            )
            env["ROS_DOMAIN_ID"] = str(domain)
        return env

    def _spawn(self, proc: Proc) -> Proc:
        self.procs.append(proc)
        return proc.start()

    def _ros_router(self, name: str, port: int) -> Proc:
        env = self._env()
        env["ZENOH_CONFIG_OVERRIDE"] = (
            f'listen/endpoints=["tcp/127.0.0.1:{port}"];connect/endpoints=[];'
            "scouting/multicast/enabled=false;plugins={}"
        )
        return self._spawn(Proc(name, [RMW_ZENOHD], env, self.tmp))

    def fleetlink_router_proc(self) -> Proc:
        env = self._env()
        # zenoh 1.8 moves payloads > 3 kB through shared memory between processes
        # on one host (transport/shared_memory/transport_optimization), which would
        # bypass the FaultProxy; a real radio link never can.  bin/fleetlink-router
        # appends its port override to this.
        env["ZENOH_CONFIG_OVERRIDE"] = "transport/shared_memory/enabled=false"
        port = self.ports["fleetlink"]
        cmd = ["bash", "-c", f"source /opt/ros/jazzy/setup.bash; exec {BIN}/fleetlink-router {port}"]  # fmt: skip
        return Proc("fleetlink_router", cmd, env, self.tmp)

    def _wait_listen(self, port: int, proc: Proc, timeout: float = 15.0) -> None:
        if not wait(lambda: port_open(port) or not proc.alive(), timeout):
            raise RuntimeError(f"{proc.name} not listening on {port}")
        if not proc.alive():
            raise RuntimeError(f"{proc.name} exited:\n{proc.log_text()[-3000:]}")

    # -- lifecycle ---------------------------------------------------------------

    def start(self, timeout: float = 60.0) -> "Fleet":
        for p in self.ports.values():
            if port_open(p):
                raise PortBusy(f"port {p} already in use")
        robots_yaml = "".join(
            f"  {r}:\n    role: robot\n    platform_type: spot\n"
            f"    addresses: {{lo: 127.0.0.1}}\n"
            for r in self.robot_names
        )
        with open(self.topology, "w") as f:
            f.write(TOPOLOGY.format(ros_port=self.ports["base_ros"], fl_port=self.ports["fleetlink"], base=self.base_name, robots=robots_yaml))  # fmt: skip

        routers = [(self._ros_router("base_router", self.ports["base_ros"]), self.ports["base_ros"])]  # fmt: skip
        self.flrouter = self._spawn(self.fleetlink_router_proc())
        routers.append((self.flrouter, self.ports["fleetlink"]))
        for r in self.robot_names:
            rr = self._ros_router(f"{r}_router", self.ports[f"{r}_ros"])
            self.robots[r]["router"] = rr
            routers.append((rr, self.ports[f"{r}_ros"]))
        for proc, port in routers:
            self._wait_listen(port, proc)

        for r in self.robot_names:
            px = FaultProxy(self.ports[f"{r}_proxy"], self.ports["fleetlink"], name=f"{r}_proxy")  # fmt: skip
            self.proxies.append(px.start())
            self.robots[r]["proxy"] = px

        benv = self._env(self.ports["base_ros"], self.domains["base"])
        fl_ep = f"tcp/127.0.0.1:{self.ports['fleetlink']}"
        self.bridge = self._spawn(Proc("bridge", [
            sys.executable, os.path.join(BIN, "fleetlink-bridge"), "--ros-args",
            "-p", f"zenoh_connect:={fl_ep}", "-p", f"topology:={self.topology}",
            "-p", f"base_name:={self.base_name}",
        ], benv, self.tmp))  # fmt: skip
        self.base = self._spawn(Helper("base_helper", [
            sys.executable, NODE_HOST, "--role", "base", "--robots",
            ",".join(self.robot_names), "--base", self.base_name, "--harness-tag", self.tag,
        ], benv, self.tmp))  # fmt: skip

        for i, r in enumerate(self.robot_names):
            renv = self._env(self.ports[f"{r}_ros"], self.domains[r])
            px = self.robots[r]["proxy"]
            self.robots[r]["agent"] = self._spawn(Proc(f"{r}_agent", [
                sys.executable, os.path.join(BIN, "fleetlink-agent"), "--ros-args",
                "-r", f"__ns:=/{r}", "-p", f"robot_name:={r}",
                "-p", "relay_config:=default", "-p", f"topology:={self.topology}",
                "-p", f"zenoh_connect:={px.endpoint}", "-p", "grid_period_s:=0.2",
            ], renv, self.tmp))  # fmt: skip
            self.robots[r]["helper"] = self._spawn(Helper(f"{r}_helper", [
                sys.executable, NODE_HOST, "--role", "robot", "--robot", r,
                "--exec-delay", str(self.exec_delay), "--tf-x", str(float(i + 1)),
                "--harness-tag", self.tag,
            ], renv, self.tmp))  # fmt: skip

        self._start_observer()
        self.wait_ready(timeout)
        return self

    def _start_observer(self) -> None:
        from fleetlink import codec
        from fleetlink.transport import ZenohTransport

        self._obs = ZenohTransport(f"tcp/127.0.0.1:{self.ports['fleetlink']}")

        def on_state(key: str, data: bytes) -> None:
            robot = key.split("/")[1]
            try:
                codec.unpack(data)
            except Exception:
                return
            self.state_times[robot].append(time.time())

        self._obs.subscribe("adt4/*/state", on_state)

        # like the console's receiver: zenoh only sends what someone subscribes to
        def on_bulk(key: str, data: bytes) -> None:
            robot = key.split("/")[1]
            self.bulk_bytes[robot] += len(data)
            if "/grid/" in key:
                self.grid_rx[robot].append((time.time(), len(data)))

        for k in ("adt4/*/grid/**", "adt4/*/mesh/**", "adt4/*/dsg/**"):
            self._obs.subscribe(k, on_bulk)

    def wait_ready(self, timeout: float) -> None:
        """Every process up, C1 flowing, and one plan round trip per robot."""
        deadline = time.time() + timeout
        for h in [self.base] + [self.robots[r]["helper"] for r in self.robot_names]:
            if h.wait_for(lambda e: e.get("ev") == "ready", max(1, deadline - time.time())) is None:  # fmt: skip
                raise RuntimeError(f"{h.name} not ready:\n{h.log_text()[-3000:]}")
        for r in self.robot_names:
            if not wait(lambda r=r: self.state_times[r], max(1, deadline - time.time())):
                raise RuntimeError(f"no C1 state from {r}:\n{self.robots[r]['agent'].log_text()[-3000:]}")  # fmt: skip
        for r in self.robot_names:
            lat = self.plan_round_trip(r, f"warm-{r}", max(1, deadline - time.time()), retry=True)  # fmt: skip
            if lat is None:
                raise RuntimeError(f"warm-up plan round trip to {r} failed:\n" + self.dump())  # fmt: skip

    def stop(self) -> None:
        if self._obs is not None:
            try:
                self._obs.close()
            except Exception:
                pass
            self._obs = None
        for p in reversed(self.procs):
            try:
                p.stop()
            except Exception:
                pass
        for px in self.proxies:
            px.stop()
        # anything a killed group leader left behind
        for _ in range(20):
            pids = stray_pids(self.tag)
            if not pids:
                break
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.1)

    # -- operations --------------------------------------------------------------

    def plan_round_trip(self, robot: str, plan_id: str, timeout: float = 10.0,
                        retry: bool = False) -> float | None:  # fmt: skip
        """Publish a plan on the base; seconds until its action_result is back on
        the base (None on timeout).  ``retry`` re-publishes (new id suffix) every
        2 s until the robot gets one: only for warm-up, while discovery settles."""
        h = self.robots[robot]["helper"]
        mb, mr = self.base.mark(), h.mark()
        t0 = time.time()
        deadline = t0 + timeout
        pid, k = plan_id, 0
        while True:
            self.base.send(cmd="plan", robot=robot, plan_id=pid)
            got = h.wait_for(lambda e: e.get("ev") == "plan" and e.get("plan_id", "").startswith(plan_id), min(2.0, max(0.1, deadline - time.time())) if retry else max(0.1, deadline - time.time()), since=mr)  # fmt: skip
            if got is not None or not retry or time.time() >= deadline:
                break
            k += 1
            pid = f"{plan_id}.{k}"
        if got is None:
            return None
        res = self.base.wait_for(lambda e: e.get("ev") == "result" and e.get("robot") == robot and e.get("plan_id") == got["plan_id"], max(0.1, deadline - time.time()), since=mb)  # fmt: skip
        return None if res is None else res["rx"] - t0

    def camera_query(self, robot: str, camera: str = "rgb", timeout: float = 10.0,
                     **params) -> tuple[float | None, dict | None, int]:  # fmt: skip
        """A C4 query as the console sends it, from the observer on the fleetlink
        router: (seconds to the reply or None, the decoded reply, reply bytes)."""
        from fleetlink import media

        done = threading.Event()
        out: dict = {}
        t0 = time.time()

        def cb(data):
            out["data"], out["t"] = data, time.time()
            done.set()

        self._obs.query(f"adt4/{robot}/media/camera/{camera}", params or None, cb, timeout=timeout)  # fmt: skip
        done.wait(timeout + 2.0)
        data = out.get("data")
        if data is None:
            return None, None, 0
        return out["t"] - t0, media.unpack_reply(data), len(data)

    def restart_fleetlink_router(self, down_s: float = 2.0, sig=signal.SIGKILL) -> None:
        self.flrouter.stop(sig=sig)
        wait(lambda: not port_open(self.ports["fleetlink"]), 5)
        time.sleep(down_s)
        self.flrouter.start()
        self._wait_listen(self.ports["fleetlink"], self.flrouter)

    def logs(self, *names: str) -> str:
        return "\n".join(p.log_text() for p in self.procs if not names or p.name in names)

    def dump(self, tail: int = 2500) -> str:
        out = []
        for p in self.procs:
            out.append(f"----- {p.name} ({'alive' if p.alive() else 'dead'})\n{p.log_text()[-tail:]}")  # fmt: skip
        for px in self.proxies:
            out.append(f"----- {px.name}: accepted={px.accepted} bytes={dict(px.bytes)} events={px.events[-10:]}")  # fmt: skip
        return "\n".join(out)

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.stop()
            raise

    def __exit__(self, *exc):
        self.stop()


class PortBusy(RuntimeError):
    pass


def max_gap(times: list[float], t0: float, t1: float) -> float:
    """Largest gap between arrivals inside [t0, t1] (window edges included)."""
    pts = [t0] + [t for t in times if t0 <= t <= t1] + [t1]
    return max(b - a for a, b in zip(pts, pts[1:]))
