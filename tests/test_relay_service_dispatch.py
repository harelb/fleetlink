"""Service dispatch deadlines, without ROS nodes, routers or real timers."""

import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from fleetlink import relay_ros


@pytest.fixture
def dispatch(monkeypatch):
    now = [10.0]
    timers, queued, replies = [], [], []

    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

        def fire(self):
            # Also exercise a callback racing with cancellation.
            self.callback()

    class Thread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    class Client:
        def __init__(self):
            self.calls = []
            self.future = Future()
            self.error = None
            self.available = True

        def wait_for_service(self, timeout_sec):
            return self.available

        def call_async(self, request):
            if self.error:
                raise self.error
            self.calls.append(request)
            return self.future

    def deserialize(data, cls):
        if data == b"invalid":
            raise ValueError("invalid request")
        return data

    monkeypatch.setitem(
        sys.modules,
        "rclpy.serialization",
        SimpleNamespace(
            deserialize_message=deserialize, serialize_message=lambda data: data
        ),
    )
    monkeypatch.setattr(
        relay_ros,
        "threading",
        SimpleNamespace(Timer=Timer, Thread=Thread, Lock=threading.Lock),
    )
    monkeypatch.setattr(relay_ros, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(
        relay_ros, "import_type", lambda name: SimpleNamespace(Request=bytes)
    )
    endpoint = relay_ros._RelayEnd.__new__(relay_ros._RelayEnd)
    client = Client()
    endpoint._clients = {"/stand": client}
    endpoint._marshal = queued.append

    def start(request=b"request"):
        endpoint._real_call(
            SimpleNamespace(type="test/srv/Stand", timeout_s=5.0),
            "/stand",
            request,
            lambda data, err=None: replies.append((data, err)),
        )

    return SimpleNamespace(
        start=start,
        client=client,
        now=now,
        timers=timers,
        queued=queued,
        replies=replies,
    )


@pytest.mark.parametrize("timer_fired", [False, True])
def test_expired_queued_call_never_reaches_service(dispatch, timer_fired):
    dispatch.start()
    dispatch.now[0] += 5.0
    if timer_fired:
        dispatch.timers[0].fire()
    dispatch.queued.pop()()
    dispatch.timers[0].fire()
    assert dispatch.client.calls == []
    assert dispatch.replies == [(None, "timeout")]


@pytest.mark.parametrize("already_complete", [False, True])
def test_service_success_replies_once(dispatch, already_complete):
    if already_complete:
        dispatch.client.future.set_result(b"response")
    dispatch.start()
    dispatch.queued.pop()()
    if not already_complete:
        dispatch.client.future.set_result(b"response")
    dispatch.timers[0].fire()
    assert dispatch.client.calls == [b"request"]
    assert dispatch.replies == [(b"response", None)]
    assert dispatch.timers[0].cancelled


def test_service_already_dispatched_can_timeout_without_second_reply(dispatch):
    dispatch.start()
    dispatch.queued.pop()()
    dispatch.timers[0].fire()
    dispatch.client.future.set_result(b"late response")
    assert dispatch.client.calls == [b"request"]
    assert dispatch.replies == [(None, "timeout")]


@pytest.mark.parametrize("where", ["request", "dispatch", "response", "unavailable"])
def test_service_failures_complete_once(dispatch, where):
    if where == "dispatch":
        dispatch.client.error = RuntimeError("dispatch failed")
    if where == "response":
        dispatch.client.future.set_exception(RuntimeError("response failed"))
    if where == "unavailable":
        dispatch.client.available = False
    dispatch.start(b"invalid" if where == "request" else b"request")
    for callback in dispatch.queued:
        callback()
    dispatch.timers[0].fire()
    assert len(dispatch.replies) == 1
    data, error = dispatch.replies[0]
    assert data is None and error != "timeout"
    assert dispatch.timers[0].cancelled
