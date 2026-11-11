"""fleetlink router (bin/fleetlink-router + config/fleetlink_router.json5) and how a
zenoh client treats a list of connect endpoints.

The endpoint-list tests pin the failover decision in fleetlink mode: the robot agent
gets *every* fleetlink endpoint of the base (one per network, in
``zenoh.fleetlink_networks`` order). A zenoh 1.8 client opens a single session to the
first endpoint that answers (never one per endpoint, so two networks reaching the same
router do not duplicate traffic), and after losing it reconnects to the first one that
answers again, without failing back while that link lasts.

Throwaway routers on free ports only; never 7447 (the live router).
"""

import json
import os
import signal
import subprocess
import threading
import time

import pytest
from conftest import _port_open

pytestmark = pytest.mark.integration

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTER_BIN = os.path.join(HERE, "..", "bin", "fleetlink-router")
ROUTER_CONFIG = os.path.join(HERE, "..", "config", "fleetlink_router.json5")
PORT_A = 17460  # NEVER 7447
PORT_B = 17461


class FleetlinkRouter:
    """bin/fleetlink-router PORT, i.e. rmw_zenohd on fleetlink_router.json5."""

    def __init__(self, port: int):
        assert port != 7447
        self.port = port
        self.proc = None
        self.output = b""

    def start(self, timeout: float = 15.0) -> None:
        if not os.path.exists("/opt/ros/jazzy/setup.bash"):
            pytest.skip("ROS jazzy not installed (rmw_zenohd)")
        if _port_open(self.port):
            pytest.skip(f"port {self.port} already in use")
        env = {k: v for k, v in os.environ.items() if k != "ZENOH_CONFIG_OVERRIDE"}
        self.proc = subprocess.Popen(
            [
                "bash",
                "-c",
                f"source /opt/ros/jazzy/setup.bash; exec {ROUTER_BIN} {self.port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        deadline = time.time() + timeout
        while time.time() < deadline and not _port_open(self.port):
            if self.proc.poll() is not None:
                self.output = self.proc.stdout.read()
                pytest.fail(f"fleetlink-router exited: {self.output.decode()[-2000:]}")
            time.sleep(0.1)
        if not _port_open(self.port):
            self.stop(kill=True)
            pytest.fail("fleetlink-router did not start listening")

    def stop(self, kill: bool = False) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL if kill else signal.SIGINT)
            self.proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc = None


@pytest.fixture
def routers():
    made = []

    def make(port):
        r = FleetlinkRouter(port)
        made.append(r)
        r.start()
        return r

    yield make
    for r in made:
        r.stop(kill=True)


def _client(endpoints):
    zenoh = pytest.importorskip("zenoh")
    conf = zenoh.Config()
    conf.insert_json5("mode", '"client"')
    conf.insert_json5("connect/endpoints", json.dumps(endpoints))
    conf.insert_json5("scouting/multicast/enabled", "false")
    # what fleetlink.transport.ZENOH_DEFAULTS sets for the connect behaviour
    conf.insert_json5("connect/timeout_ms", "0")
    conf.insert_json5("connect/exit_on_failure", "false")
    conf.insert_json5(
        "connect/retry",
        '{"period_init_ms": 500, "period_max_ms": 1000, "period_increase_factor": 2}',
    )
    return zenoh.open(conf)


def _router_conns(ports):
    """ESTABLISHED tcp connections of this process to the given remote ports."""
    psutil = pytest.importorskip("psutil")
    proc = psutil.Process()
    conns = getattr(proc, "net_connections", None) or proc.connections
    return sorted(
        c.raddr.port
        for c in conns(kind="tcp")
        if c.status == "ESTABLISHED" and c.raddr and c.raddr.port in ports
    )


def _wait(pred, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


class _Sub:
    def __init__(self, session, key):
        self.got = []
        self.lock = threading.Lock()
        self.sub = session.declare_subscriber(key, self._cb)

    def _cb(self, sample):
        with self.lock:
            self.got.append(sample.payload.to_bytes())

    def values(self):
        with self.lock:
            return list(self.got)


def _close(*sessions):
    for s in sessions:
        if s is None:
            continue
        try:
            s.close()
        except Exception:
            pass


def test_router_config_is_minimal_and_valid():
    zenoh = pytest.importorskip("zenoh")
    conf = zenoh.Config.from_file(ROUTER_CONFIG)  # parses with zenoh 1.8
    assert json.loads(conf.get_json("mode")) == "router"
    assert json.loads(conf.get_json("listen/endpoints")) == ["tcp/[::]:7450"]
    assert json.loads(conf.get_json("connect/endpoints")) == []
    assert json.loads(conf.get_json("scouting/multicast/enabled")) is False
    assert json.loads(conf.get_json("scouting/gossip/enabled")) is False
    assert json.loads(conf.get_json("adminspace/enabled")) is False


def test_router_data_queues_hold_large_messages():
    """The router fragments what it forwards with 8 kB batches (negotiated with the
    agent): zenoh's default 2-batch queue drops any DROP message over ~16 kB
    whenever it is busy.  The data priorities get zenoh's maximum, 16."""
    zenoh = pytest.importorskip("zenoh")
    conf = zenoh.Config.from_file(ROUTER_CONFIG)
    size = json.loads(conf.get_json("transport/link/tx/queue/size"))
    assert {k: size[k] for k in ("data_high", "data", "data_low")} == dict.fromkeys(
        ("data_high", "data", "data_low"), 16
    )
    assert size["interactive_high"] == 2  # untouched: zenoh default


def test_wrapper_refuses_7447_and_bad_ports():
    for arg in ("7447", "abc", "70000"):
        r = subprocess.run(
            [ROUTER_BIN, arg], capture_output=True, text=True, timeout=10
        )
        assert r.returncode == 2, (arg, r.stdout, r.stderr)


def test_fleetlink_router_starts_and_relays_between_clients(routers):
    routers(PORT_A)
    ep = f"tcp/127.0.0.1:{PORT_A}"
    a = b = None
    try:
        a, b = _client([ep]), _client([ep])
        sub = _Sub(b, "adt4/test/**")
        time.sleep(0.5)
        assert _wait(lambda: (a.put("adt4/test/x", b"hi"), b"hi" in sub.values())[1], 5)
    finally:
        _close(a, b)


def test_two_endpoints_same_router_one_session_no_duplicates(routers):
    """Both endpoints reach the same router: one session, each message once."""
    routers(PORT_A)
    rx = tx = None
    try:
        rx = _client([f"tcp/127.0.0.1:{PORT_A}"])
        sub = _Sub(rx, "adt4/dup/**")
        before = _router_conns({PORT_A})
        tx = _client([f"tcp/127.0.0.1:{PORT_A}", f"tcp/[::1]:{PORT_A}"])
        time.sleep(1.0)
        assert len(_router_conns({PORT_A})) == len(before) + 1  # one link, not two
        n = 30
        for i in range(n):
            tx.put("adt4/dup/x", str(i).encode())
            time.sleep(0.01)
        _wait(lambda: len(sub.values()) >= n, 5)
        time.sleep(0.5)
        assert sorted(int(v) for v in sub.values()) == list(range(n))
    finally:
        _close(tx, rx)


def test_endpoint_list_is_ordered_failover(routers):
    """[A, B]: connects to A only; A dies -> B; A back -> stays on B."""
    ra = routers(PORT_A)
    routers(PORT_B)
    rx_b = tx = None
    try:
        rx_b = _client([f"tcp/127.0.0.1:{PORT_B}"])
        sub_b = _Sub(rx_b, "adt4/fo/**")
        base_b = _router_conns({PORT_B})
        tx = _client([f"tcp/127.0.0.1:{PORT_A}", f"tcp/127.0.0.1:{PORT_B}"])
        time.sleep(1.0)
        assert _router_conns({PORT_A, PORT_B}) == sorted([PORT_A] + base_b)
        ra.stop()
        assert _wait(lambda: _router_conns({PORT_A, PORT_B}).count(PORT_B) == 2, 15)
        assert _wait(
            lambda: (tx.put("adt4/fo/x", b"via-b"), b"via-b" in sub_b.values())[1], 5
        )
        ra.start()
        time.sleep(3.0)
        assert PORT_A not in _router_conns({PORT_A, PORT_B})  # no fail-back
    finally:
        _close(tx, rx_b)


def test_dead_first_endpoint_falls_through(routers):
    """A refused first endpoint (router down) costs nothing: B is used at once."""
    routers(PORT_B)
    tx = None
    try:
        t0 = time.time()
        tx = _client([f"tcp/127.0.0.1:{PORT_A}", f"tcp/127.0.0.1:{PORT_B}"])
        assert time.time() - t0 < 5
        assert _wait(lambda: PORT_B in _router_conns({PORT_A, PORT_B}), 5)
    finally:
        _close(tx)
