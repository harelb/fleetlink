import os
import signal
import socket
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

ROUTER_PORT = 17448  # throwaway router; NEVER 7447 (the live system's router)
ROUTER_ENDPOINT = f"tcp/127.0.0.1:{ROUTER_PORT}"
LATE_ROUTER_PORT = 17449  # started after the sessions (no-router / restart tests)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs a zenoh router / ROS / real data"
    )


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


class Router:
    """A throwaway rmw_zenohd listening on 127.0.0.1:<port> (NEVER 7447)."""

    def __init__(self, port: int):
        assert port != 7447, "7447 is the live system's router"
        self.port = port
        self.endpoint = f"tcp/127.0.0.1:{port}"
        self.proc = None

    def start(self, timeout: float = 15.0) -> None:
        """Start it and wait until it listens; pytest.skip when it cannot."""
        if not os.path.exists("/opt/ros/jazzy/setup.bash"):
            pytest.skip("ROS jazzy not installed (rmw_zenohd)")
        if _port_open(self.port):
            pytest.skip(f"port {self.port} already in use")
        override = (
            f'listen/endpoints=["{self.endpoint}"];connect/endpoints=[];'
            "scouting/multicast/enabled=false;adminspace/enabled=false;plugins={}"
        )
        cmd = (
            "source /opt/ros/jazzy/setup.bash; "
            f"ZENOH_CONFIG_OVERRIDE='{override}' exec ros2 run rmw_zenoh_cpp rmw_zenohd"
        )
        self.proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + timeout
        while time.time() < deadline and not _port_open(self.port):
            if self.proc.poll() is not None:
                pytest.skip("rmw_zenohd exited during startup")
            time.sleep(0.1)
        if not _port_open(self.port):
            self.stop(kill=True)
            pytest.skip("rmw_zenohd did not start listening")

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


@pytest.fixture(scope="session")
def zenoh_router():
    """A throwaway rmw_zenohd on 127.0.0.1:17448 (killed at teardown)."""
    router = Router(ROUTER_PORT)
    router.start()
    yield router.endpoint
    router.stop()


@pytest.fixture
def late_router():
    """A router on 127.0.0.1:17449 that the test starts / stops itself."""
    router = Router(LATE_ROUTER_PORT)
    if _port_open(LATE_ROUTER_PORT):
        pytest.skip(f"port {LATE_ROUTER_PORT} already in use")
    yield router
    router.stop(kill=True)
