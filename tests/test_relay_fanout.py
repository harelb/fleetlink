"""Robot-local uplinks can feed one global topic and fan out without echo loops."""

import threading
from types import SimpleNamespace as NS

import pytest

from fleetlink import relay_ros
from fleetlink.relay import RelayConfig, RelayConfigError, RelayCore
from fleetlink.transport import CLASSES, LinkEmulator, VirtualClock


def config():
    return RelayConfig(
        {
            "topics": [
                {
                    "topic": "/{robot}/progress",
                    "base_topic": "/progress",
                    "type": "pkg/msg/Progress",
                    "dir": "up",
                    "mode": "reliable",
                },
                {
                    "topic": "/progress",
                    "type": "pkg/msg/Progress",
                    "dir": "down",
                    "mode": "reliable",
                },
            ]
        }
    )


@pytest.mark.parametrize(
    "alias,direction", [("relative", "up"), ("/progress", "down"), (123, "up")]
)
def test_invalid_base_publication_names_are_rejected(alias, direction):
    with pytest.raises(RelayConfigError, match="base_topic"):
        RelayConfig(
            {
                "topics": [
                    {
                        "topic": "/a",
                        "base_topic": alias,
                        "type": "p/msg/M",
                        "dir": direction,
                        "mode": "reliable",
                    }
                ]
            }
        )


def test_uplink_is_published_on_base_and_reaches_every_robot_once(monkeypatch):
    monkeypatch.setattr(relay_ros, "import_type", lambda _: bytes)
    monkeypatch.setattr(relay_ros, "qos_profile", lambda qos: qos)
    monkeypatch.setattr(relay_ros, "type_hash", lambda _: "same-type")
    clock = VirtualClock(1000.0)
    link = LinkEmulator(
        clock=clock,
        bandwidth=None,
        latency=0.01,
        reliable=dict.fromkeys(CLASSES, False),
    )
    callbacks, publications = {}, []

    class Node:
        def create_subscription(self, cls, topic, callback, *args, **kwargs):
            callbacks.setdefault(topic, []).append(callback)

        def create_publisher(self, cls, topic, qos):
            def publish(data):
                publications.append((topic, data))
                for callback in callbacks.get(topic, []):
                    callback(data)

            return NS(
                publish=publish,
                get_subscription_count=lambda: len(callbacks.get(topic, [])),
            )

    bridge = relay_ros.BaseBridge.__new__(relay_ros.BaseBridge)
    bridge.node = Node()
    bridge.config = config()
    bridge.base = "base"
    bridge.transport = link.base
    bridge._lock = threading.Lock()
    bridge._cores, bridge._pubs = {}, {}
    bridge._on_event = lambda *_: None
    bridge.log = NS(info=lambda *_: None, warn=lambda *_: None)
    bridge._publish = lambda pub, data: (
        pub.publish(data),
        pub.get_subscription_count(),
    )[1]
    received = {"a": [], "b": []}
    robots = []
    for name in received:
        bridge.add_robot(name, "spot")
        robots.append(
            RelayCore(
                name,
                "robot",
                link.robot,
                bridge.config,
                platform="spot",
                type_hash=lambda _: "same-type",
                clock=clock,
                on_message=lambda e, data, meta, r=name: received[r].append(
                    (e.topic, data)
                ),
            )
        )
    # The bridge's real clock must share the deterministic transport's clock.
    for core in bridge.cores():
        core.clock = clock
    for _ in range(20):
        for core in [*robots, *bridge.cores()]:
            core.tick()
        clock.advance(0.05)
        link.advance()
    assert robots[0].publish("/a/progress", b"DONE:a:task1")
    for _ in range(80):
        for core in [*robots, *bridge.cores()]:
            core.tick()
        clock.advance(0.05)
        link.advance()
    assert publications == [("/progress", b"DONE:a:task1")]
    assert received == {
        "a": [("/progress", b"DONE:a:task1")],
        "b": [("/progress", b"DONE:a:task1")],
    }
