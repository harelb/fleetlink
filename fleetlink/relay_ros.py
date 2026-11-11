"""Generic ROS relay over fleetlink: the rclpy half (see ``relay.py`` for the wire).

* :class:`RobotRelay` runs inside the robot's ``fleetlink-agent`` (``AgentNode``,
  param ``relay_config``; off by default) and shares its zenoh transport.  It
  subscribes the ``up`` topics on the robot graph, republishes ``down`` topics
  there, composes the configured transforms from a tf2 buffer, serves proxy
  servers for ``dir: up`` services (the robot calls, the base answers) and calls
  the robot's real servers for ``dir: down`` services.
* :class:`BaseBridge` is the ``fleetlink-bridge`` node on the base: one process
  for all robots (from the topology, plus any robot whose relay heartbeat
  arrives).  Per robot it subscribes the ``down`` topics on the base graph,
  republishes ``up`` topics under the same names, publishes the robots' transforms
  on ``/tf`` and ``/tf_static`` (all robots' static transforms in one latched
  message), serves proxies for ``dir: down`` services and calls the real base
  servers (``base_name``) for ``dir: up`` ones, and publishes one
  ``std_msgs/String`` JSON per robot on ``fleetlink/link_status`` (1 Hz, and
  right after a ``down`` message is acked): ``RelayCore.status()``.

Messages cross as CDR bytes (``raw=True`` subscriptions, ``publish(bytes)``), so
the relay never deserialises them, except TF (composed / aggregated).  All
publishing and service-client calls are marshalled onto the executor thread
(rmw_zenoh has had heap corruption publishing from foreign threads), and so is
the creation of every rclpy entity (service clients are created up front).

Proxy service servers block an executor thread in a reentrant callback group
while the far side answers: spin the node with ``make_executor()`` (a
``MultiThreadedExecutor`` of :data:`EXECUTOR_THREADS`).  At most
:data:`MAX_PROXY_CALLS` relayed calls per end wait at once, and a call made while
the link is down fails at once (default response, warning logged), so proxies
never take the threads the DSG / TF / pose callbacks need.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
import pathlib
import queue
import threading
import time

from fleetlink.relay import RelayConfig, RelayCore

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RELAY_CONFIG = HERE.parent / "config" / "relay.yaml"
DEFAULT_TOPOLOGY = HERE.parents[1] / "network_diagnostics" / "config" / "topology.yaml"
DEFAULT_PLATFORM = "spot"  # robots without platform_type in the topology
# executor threads of fleetlink-agent / fleetlink-bridge (rclpy's default is the CPU
# count), and the relayed service calls one end lets wait at once: the rest of the
# threads always stay free for the node's own callbacks
EXECUTOR_THREADS = 8
MAX_PROXY_CALLS = 2


def make_executor(context=None):
    from rclpy.executors import MultiThreadedExecutor

    return MultiThreadedExecutor(num_threads=EXECUTOR_THREADS, context=context)


def import_type(path: str):
    """``"pkg/msg/Name"`` / ``"pkg/srv/Name"`` → class."""
    pkg, kind, name = path.split("/")
    return getattr(importlib.import_module(f"{pkg}.{kind}"), name)


@functools.lru_cache(maxsize=None)
def type_hash(type_str: str) -> str | None:
    """The ROS 2 type hash (``RIHS01_…``, from the package's generated
    ``<Name>.json``, Jazzy+); else ``SHA256_…`` of the interface definition file;
    else of the class's field layout; None when the type is unknown."""
    try:
        pkg, kind, name = type_str.split("/")
    except ValueError:
        return None
    share = None
    try:
        from ament_index_python.packages import get_package_share_directory

        share = pathlib.Path(get_package_share_directory(pkg))
    except Exception:
        pass
    if share is not None:
        js = share / kind / f"{name}.json"
        if js.is_file():
            try:
                for h in json.loads(js.read_text()).get("type_hashes", []):
                    if h.get("type_name") == type_str:
                        return h["hash_string"]
            except Exception:
                pass
        for ext in (".msg", ".srv", ".idl"):
            f = share / kind / f"{name}{ext}"
            if f.is_file():
                return "SHA256_" + hashlib.sha256(f.read_bytes()).hexdigest()
    try:
        cls = import_type(type_str)
    except Exception:
        return None
    parts = [cls] if hasattr(cls, "get_fields_and_field_types") else []
    parts += [getattr(cls, n) for n in ("Request", "Response") if hasattr(cls, n)]
    text = repr([p.get_fields_and_field_types() for p in parts])
    return "SHA256_" + hashlib.sha256(text.encode()).hexdigest()


def qos_profile(q: dict):
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(q.get("depth") or 10),
        reliability=ReliabilityPolicy.BEST_EFFORT
        if q.get("reliability") == "best_effort"
        else ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL
        if q.get("durability") == "transient_local"
        else DurabilityPolicy.VOLATILE,
    )


def load_topology(path) -> dict:
    import yaml

    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


PLATFORM_ALIASES = {"husky": "phoenix"}


def platform_of(machine: dict | None) -> str:
    """A topology machine's platform: ``platform_type``, missing → spot, husky → phoenix.

    The one rule the agent, bridge, console and fleet tooling all use.
    """
    p = str((machine or {}).get("platform_type") or DEFAULT_PLATFORM).lower()
    return PLATFORM_ALIASES.get(p, p)


def topology_robots(topology: dict) -> dict[str, str]:
    """``{robot: platform_type}`` for every ``role: robot`` machine."""
    out = {}
    for name, m in ((topology or {}).get("machines") or {}).items():
        if (m or {}).get("role") == "robot":
            out[name] = platform_of(m)
    return out


class _Marshal:
    """Run callables on the node's executor (a queue drained by a guard condition)."""

    def __init__(self, node):
        self._q: queue.Queue = queue.Queue()
        self._log = node.get_logger()
        self._guard = node.create_guard_condition(self._drain)

    def __call__(self, fn) -> None:
        self._q.put(fn)
        self._guard.trigger()

    def _drain(self) -> None:
        while True:
            try:
                fn = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self._log.error(f"fleetlink relay: executor call failed: {exc}")


class _RelayEnd:
    """What both ends share: raw publishers, proxy service servers, real service
    calls, the tick thread."""

    def __init__(self, node, tick_hz: float = 20.0, max_proxy_calls: int = MAX_PROXY_CALLS):
        from rclpy.callback_groups import ReentrantCallbackGroup

        self.node = node
        self.log = node.get_logger()
        self._marshal = _Marshal(node)
        self._cbg = ReentrantCallbackGroup()
        self._clients: dict[str, object] = {}
        self._proxy_slots = threading.BoundedSemaphore(max(1, int(max_proxy_calls)))
        self.proxy_refused = 0  # relayed calls failed at once (link down / busy)
        self._running = True
        self._tick_period = 1.0 / tick_hz
        self._ticker = threading.Thread(
            target=self._tick_loop, daemon=True, name="fleetlink-relay"
        )

    def cores(self) -> list[RelayCore]:
        raise NotImplementedError

    def _tick_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            now = time.time()
            for core in self.cores():
                try:
                    core.tick(now)
                except Exception as exc:  # noqa: BLE001
                    self.log.error(f"relay tick failed: {exc}")
            self._after_tick(now)
            time.sleep(max(0.0, self._tick_period - (time.monotonic() - t0)))

    def _after_tick(self, now: float) -> None:
        pass

    def _publish(self, pub, data: bytes) -> int:
        self._marshal(lambda: pub.publish(bytes(data)))
        return pub.get_subscription_count()

    def _proxy_server(self, core: RelayCore, s, ros_name: str):
        """A local server whose requests are relayed to the peer's real server."""
        from rclpy.serialization import deserialize_message, serialize_message

        cls = import_type(s.type)

        def refuse(why: str, response):
            self.proxy_refused += 1
            self.log.warn(f"relayed call {ros_name} failed at once: {why}")
            return response

        def handle(request, response):
            if not core.connected():
                return refuse(f"no link to {core.robot}'s peer", response)
            if not self._proxy_slots.acquire(blocking=False):
                return refuse("too many relayed calls in progress", response)
            try:
                done = threading.Event()
                box: dict = {}

                def finish(data, err):
                    box["data"], box["err"] = data, err
                    done.set()

                core.call(s.name, serialize_message(request), finish)
                done.wait(s.timeout_s + 2.0)
            finally:
                self._proxy_slots.release()
            if box.get("data") is not None:
                return deserialize_message(box["data"], cls.Response)
            self.log.warn(
                f"relayed call {ros_name} failed: {box.get('err', 'timeout')}"
            )
            return response

        return self.node.create_service(cls, ros_name, handle, callback_group=self._cbg)

    def _real_call(self, s, ros_name: str, request: bytes, reply) -> None:
        """Call the real local server ``ros_name``; ``reply(bytes | None, err)``."""
        from rclpy.serialization import deserialize_message, serialize_message

        cls = import_type(s.type)
        timer = threading.Timer(s.timeout_s, lambda: reply(None, "timeout"))
        timer.daemon = True
        timer.start()

        def done(fut):
            timer.cancel()
            try:
                reply(serialize_message(fut.result()), None)
            except Exception as exc:  # noqa: BLE001
                reply(None, f"{type(exc).__name__}: {exc}")

        def work():
            client = self._client(cls, ros_name)
            if client is None:
                timer.cancel()
                reply(None, f"no client for {ros_name}")
                return
            if not client.wait_for_service(timeout_sec=min(5.0, s.timeout_s)):
                timer.cancel()
                reply(None, f"service {ros_name} not available")
                return
            req = deserialize_message(request, cls.Request)

            def call():
                client.call_async(req).add_done_callback(done)

            self._marshal(call)

        threading.Thread(target=work, daemon=True, name="fleetlink-srv").start()

    def _create_client(self, cls, ros_name: str):
        """Executor thread (or before spinning)."""
        client = self._clients.get(ros_name)
        if client is None:
            client = self._clients[ros_name] = self.node.create_client(
                cls, ros_name, callback_group=self._cbg
            )
        return client

    def _client(self, cls, ros_name: str, timeout: float = 5.0):
        """The client of ``ros_name``; one missing is created on the executor
        thread (never on zenoh's or this worker thread)."""
        client = self._clients.get(ros_name)
        if client is not None:
            return client
        done = threading.Event()

        def create():
            try:
                self._create_client(cls, ros_name)
            finally:
                done.set()

        self._marshal(create)
        done.wait(timeout)
        return self._clients.get(ros_name)

    def shutdown(self) -> None:
        self._running = False
        if self._ticker.is_alive():
            self._ticker.join(timeout=2.0)
        for core in self.cores():
            core.close()


class RobotRelay(_RelayEnd):
    """Robot side; attach to the agent's node and transport."""

    def __init__(
        self,
        node,
        transport,
        robot: str,
        config: RelayConfig,
        platform: str | None = None,
        tf_buffer=None,
        tick_hz: float = 20.0,
    ):
        super().__init__(node, tick_hz)
        self.robot = robot
        nd = node
        # publishers first: a message may arrive as soon as the core subscribes
        self._pubs: dict[str, object] = {}
        for e in config.resolve(robot, platform).in_topics("robot"):
            self._pubs[e.name] = nd.create_publisher(
                import_type(e.type), e.topic, qos_profile(e.qos)
            )
        self.core = RelayCore(
            robot,
            "robot",
            transport,
            config,
            platform=platform,
            on_message=self._on_message,
            on_service=self._on_service,
            type_hash=type_hash,
            log=self.log.warn,
        )
        self._tf_buffer = tf_buffer
        self._tf_sent: dict[str, tuple] = {}
        self._static: dict[str, object] = {}  # child frame -> TransformStamped
        self._static_sent = None
        for e in self.core.out.values():
            e = e.entry
            if e.kind == "topic":
                nd.create_subscription(
                    import_type(e.type),
                    e.topic,
                    lambda raw, n=e.name: self.core.publish(n, raw),
                    qos_profile(e.qos),
                    raw=True,
                )
            elif e.kind == "tf":
                self._ensure_tf_buffer()
                nd.create_timer(1.0 / (e.rate_hz or 5.0), lambda e=e: self._tf_tick(e))
            elif e.kind == "tf_static":
                from tf2_msgs.msg import TFMessage

                nd.create_subscription(
                    TFMessage,
                    "/tf_static",
                    lambda m, e=e: self._on_static(e, m),
                    qos_profile({"durability": "transient_local", "depth": 100}),
                )
        self._servers = [
            self._proxy_server(self.core, s, s.name) for s in self.core.srv_out.values()
        ]
        for s in self.core.srv_in.values():  # base calls robot: the real servers
            self._create_client(import_type(s.type), s.name)
        self._ticker.start()
        self.log.info(
            f"fleetlink relay robot={robot} platform={platform} cfg={config.hash} "
            f"up={sorted(self.core.out)} down={sorted(self.core.inn)} "
            f"services={sorted(self.core.resolved.services)}"
        )

    def cores(self):
        return [self.core]

    def _ensure_tf_buffer(self):
        if self._tf_buffer is None:
            import tf2_ros

            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

    def _tf_tick(self, e) -> None:
        from rclpy.serialization import serialize_message
        from rclpy.time import Time
        from tf2_msgs.msg import TFMessage

        try:
            tf = self._tf_buffer.lookup_transform(e.parent, e.child, Time())
        except Exception:
            return
        stamp = (tf.header.stamp.sec, tf.header.stamp.nanosec)
        if self._tf_sent.get(e.name) == stamp:
            return  # nothing new
        self._tf_sent[e.name] = stamp
        self.core.publish(e.name, serialize_message(TFMessage(transforms=[tf])))

    def _on_static(self, e, msg) -> None:
        from rclpy.serialization import serialize_message
        from tf2_msgs.msg import TFMessage

        pre = e.frames_prefix
        # a frame that is the child of a composed transform (map -> base_link)
        # must not get a second parent from the static tree on the base
        composed = {
            o.entry.child for o in self.core.out.values() if o.entry.kind == "tf"
        }
        for t in msg.transforms:
            if t.child_frame_id in composed:
                continue
            if t.child_frame_id.startswith(pre) or t.header.frame_id.startswith(pre):
                self._static[t.child_frame_id] = t
        key = tuple(
            sorted(
                (c, t.header.frame_id, t.transform.translation.x, t.transform.translation.y,
                 t.transform.translation.z, t.transform.rotation.w, t.transform.rotation.x,
                 t.transform.rotation.y, t.transform.rotation.z)
                for c, t in self._static.items()
            )
        )  # fmt: skip
        if not self._static or key == self._static_sent:
            return
        self._static_sent = key
        msg = TFMessage(transforms=list(self._static.values()))
        self.core.publish(e.name, serialize_message(msg))

    def _on_message(self, e, data: bytes, meta: dict):
        pub = self._pubs.get(e.name)
        return None if pub is None else self._publish(pub, data)

    def _on_service(self, s, request: bytes, reply) -> None:
        self._real_call(s, s.name, request, reply)


class BaseBridge(_RelayEnd):
    """Base side: one node for every robot (``fleetlink-bridge``)."""

    def __init__(
        self,
        node,
        transport,
        config: RelayConfig,
        robots: dict[str, str] | None = None,
        base: str = "",
        discover: bool = True,
        tick_hz: float = 20.0,
        status_period: float = 1.0,
    ):
        super().__init__(node, tick_hz)
        from std_msgs.msg import String

        self.transport = transport
        self.config = config
        self.base = base
        self.status_period = status_period
        self._lock = threading.Lock()
        self._cores: dict[str, RelayCore] = {}
        self._pubs: dict[tuple, object] = {}  # (topic, type) -> publisher (shared)
        self._static: dict[str, dict] = {}  # robot -> child -> TransformStamped
        self._dirty: set[str] = set()
        self._t_status = 0.0
        self._String = String
        self.status_pub = node.create_publisher(String, "/fleetlink/link_status", 10)
        for r, platform in (robots or {}).items():
            self.add_robot(r, platform)
        self._discovery = None
        if discover:
            self._discovery = transport.subscribe(
                "adt4/*/ros/ack/up", self._on_discovery
            )
        self._ticker.start()
        self.log.info(
            f"fleetlink-bridge base={base} cfg={config.hash} robots={sorted(self._cores)}"
        )

    def cores(self):
        with self._lock:
            return list(self._cores.values())

    def core(self, robot: str) -> RelayCore | None:
        with self._lock:
            return self._cores.get(robot)

    # -- robots ---------------------------------------------------------------

    def _on_discovery(self, key: str, data: bytes) -> None:
        robot = key.split("/")[1]
        with self._lock:
            if robot in self._cores:
                return
        from fleetlink import codec

        try:
            p = codec.unpack(data).get("p") or {}
        except Exception:
            return
        if p.get("op") != "hb" or p.get("side") != "robot":
            return
        platform = p.get("platform") or DEFAULT_PLATFORM
        self._marshal(lambda: self.add_robot(robot, platform))

    def add_robot(self, robot: str, platform: str | None = None) -> RelayCore:
        """Executor thread (or before spinning): create the robot's endpoints."""
        with self._lock:
            if robot in self._cores:
                return self._cores[robot]
        platform = (platform or DEFAULT_PLATFORM).lower()
        nd = self.node
        # up: publish on the base graph (before the core subscribes)
        for e in self.config.resolve(robot, platform, self.base).in_topics("base"):
            k = (e.topic, e.type)
            if k not in self._pubs:
                self._pubs[k] = nd.create_publisher(
                    import_type(e.type), e.topic, qos_profile(e.qos)
                )
        core = RelayCore(
            robot,
            "base",
            self.transport,
            self.config,
            platform=platform,
            base=self.base,
            on_message=lambda e, d, m, r=robot: self._on_message(r, e, d, m),
            on_service=self._on_service,
            on_event=self._on_event,
            type_hash=type_hash,
            log=self.log.warn,
        )
        for o in core.out.values():  # down: subscribe on the base graph
            e = o.entry
            nd.create_subscription(
                import_type(e.type),
                e.topic,
                lambda raw, n=e.name, c=core: c.publish(n, raw),
                qos_profile(e.qos),
                raw=True,
            )
        for s in core.srv_out.values():  # base calls robot: proxy server here
            self._proxy_server(core, s, s.base_name)
        for s in core.srv_in.values():  # robot calls base: the real base servers
            self._create_client(import_type(s.type), s.base_name)
        with self._lock:
            self._cores[robot] = core
        self.log.info(
            f"relay robot={robot} platform={platform} up={sorted(core.inn)} "
            f"down={sorted(core.out)} services={sorted(core.resolved.services)}"
        )
        return core

    # -- inbound ----------------------------------------------------------------

    def _on_message(self, robot: str, e, data: bytes, meta: dict):
        pub = self._pubs.get((e.topic, e.type))
        if pub is None:
            return None
        if e.kind != "tf_static":
            return self._publish(pub, data)
        from rclpy.serialization import deserialize_message
        from tf2_msgs.msg import TFMessage

        msg = deserialize_message(data, TFMessage)
        with self._lock:
            self._static[robot] = {t.child_frame_id: t for t in msg.transforms}
            union = [t for d in self._static.values() for t in d.values()]
        self._marshal(lambda: pub.publish(TFMessage(transforms=union)))
        return pub.get_subscription_count()

    def _on_service(self, s, request: bytes, reply) -> None:
        self._real_call(s, s.base_name, request, reply)

    def _on_event(self, kind: str, info: dict) -> None:
        if kind == "ack":
            with self._lock:
                self._dirty.add(info["robot"])
        elif kind == "peer_restart":
            self.log.warn(
                f"robot {info['robot']} relay restarted (epoch {info['epoch']})"
            )
            with self._lock:
                self._dirty.add(info["robot"])

    # -- status -----------------------------------------------------------------

    def _after_tick(self, now: float) -> None:
        with self._lock:
            if now - self._t_status >= self.status_period:
                self._t_status = now
                robots = list(self._cores)
            else:
                robots = list(self._dirty)
            self._dirty.clear()
            cores = [self._cores[r] for r in robots if r in self._cores]
        for core in cores:
            text = json.dumps(core.status(now), default=str)
            msg = self._String(data=text)
            self._marshal(lambda m=msg: self.status_pub.publish(m))

    def shutdown(self) -> None:
        if self._discovery is not None:
            unsub = getattr(self.transport, "unsubscribe", None)
            try:
                unsub(self._discovery) if unsub else self._discovery.undeclare()
            except Exception:
                pass
        super().shutdown()


# ----------------------------------------------------------------------------
# fleetlink-bridge
# ----------------------------------------------------------------------------


def _base_name(topology: dict) -> str:
    env = os.environ.get("ADT4_ROBOT_NAME")
    if env:
        return env
    fb = ((topology or {}).get("fleet_defaults") or {}).get("fleetlink_base")
    return fb or "base_station"


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node

    from fleetlink.agent import zenoh_version_warning
    from fleetlink.transport import ZenohTransport, split_endpoints

    rclpy.init(args=args)
    node = Node("fleetlink_bridge")
    p = lambda name, default: node.declare_parameter(name, default).value  # noqa: E731
    connect = split_endpoints(p("zenoh_connect", "tcp/127.0.0.1:7450"))
    topo_path = p("topology", "") or str(DEFAULT_TOPOLOGY)
    relay_path = p("relay_config", "") or str(DEFAULT_RELAY_CONFIG)
    robots_param = p("robots", "")  # comma-separated override of the topology robots
    topology = load_topology(topo_path)
    base = p("base_name", "") or _base_name(topology)
    log = node.get_logger()
    warning = zenoh_version_warning()
    if warning:
        log.warn(warning)
    config = RelayConfig.load(relay_path)
    robots = topology_robots(topology)
    if robots_param:
        names = [r.strip() for r in robots_param.split(",") if r.strip()]
        robots = {r: robots.get(r, DEFAULT_PLATFORM) for r in names}
    transport = ZenohTransport(connect)
    if not transport.opened.is_set():
        log.warn(f"no zenoh router at {connect} yet ({transport.open_error}); retrying")
    bridge = BaseBridge(node, transport, config, robots, base=base)
    executor = make_executor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        executor.shutdown()
        transport.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
