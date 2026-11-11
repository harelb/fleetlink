"""Console-script entry points for ``fleetlink-agent`` and ``fleetlink-bridge``.

Both need ``eclipse-zenoh`` pinned to the zenoh version of ``rmw_zenoh`` (1.8),
which often lives in a separate virtualenv rather than the system Python that
``ros2 run`` / the installed scripts use. When ``FLEETLINK_PYTHON`` names
another interpreter (or, for ADT4 deployments, ``$ADT4_ENV/spark_env`` exists),
the entry point re-executes itself there (same entry point, same arguments),
passing every argument (``--ros-args ...``) through. The sourced workspace's
``PYTHONPATH`` makes the installed package importable in that interpreter.

Only the standard library is imported here, so the re-exec happens before
anything that needs the virtualenv.
"""

from __future__ import annotations

import os
import pathlib
import sys

REEXEC_GUARD = "_FLEETLINK_REEXEC"

AGENT_USAGE = """\
usage: fleetlink-agent --ros-args [-r __ns:=/<robot>] [-p name:=value ...]

Robot-side fleetlink agent: streams the robot's scene graph, mesh, state,
cameras and (optionally) image store to the base over zenoh, and runs the
generic ROS relay when relay_config is set.

Main parameters (all optional; see the README for the full list):
  robot_name      robot name (default: the node namespace)
  zenoh_connect   comma-separated zenoh endpoints (default tcp/127.0.0.1:7447)
  config          fleetlink.yaml overriding the built-in defaults
  relay_config    relay table (relay.example.yaml schema); off when empty
  topology        topology.yaml (default: $ADT4_TOPOLOGY)
  platform        spot | phoenix (default: from the topology, else spot)
  cameras         list of name=topic camera streams

Environment: FLEETLINK_PYTHON (interpreter with eclipse-zenoh), FLEETLINK_FFMPEG.
"""

BRIDGE_USAGE = """\
usage: fleetlink-bridge --ros-args -p relay_config:=<relay.yaml> [-p name:=value ...]

Base-side fleetlink bridge: relays every robot's ROS topics, transforms and
services (as listed in the relay table) into the base ROS graph.

Parameters:
  relay_config    relay table (required; schema: config/relay.example.yaml)
  topology        topology.yaml (required unless $ADT4_TOPOLOGY is set)
  zenoh_connect   fleetlink router endpoints (default tcp/127.0.0.1:7450)
  base_name       base station name (default: $ADT4_ROBOT_NAME, else the
                  topology's fleet_defaults.fleetlink_base, else base_station)
  robots          comma-separated robots (default: the topology's robots)

Environment: FLEETLINK_PYTHON (interpreter with eclipse-zenoh).
"""


def target_python() -> str | None:
    """The interpreter to run in: ``$FLEETLINK_PYTHON``, else ADT4's
    ``$ADT4_ENV/spark_env/bin/python3`` when that exists; None to stay here."""
    py = os.environ.get("FLEETLINK_PYTHON")
    if py:
        return py
    adt4_env = os.environ.get("ADT4_ENV")
    if adt4_env:
        cand = pathlib.Path(adt4_env) / "spark_env" / "bin" / "python3"
        if cand.exists():
            return str(cand)
    return None


def _same_env(py: str) -> bool:
    # a virtualenv's python is a symlink to the base interpreter, so compare the
    # environment prefix (<env>/bin/python3 -> <env>), not the resolved binary
    env = pathlib.Path(py).absolute().parent.parent
    return pathlib.Path(sys.prefix).absolute() == env


def _maybe_reexec(entry: str) -> None:
    if os.environ.get(REEXEC_GUARD):
        return
    py = target_python()
    if not py or _same_env(py):
        return
    if not os.path.exists(py):
        sys.exit(f"fleetlink: FLEETLINK_PYTHON={py} does not exist")
    os.environ[REEXEC_GUARD] = "1"
    code = f"from fleetlink.launcher import {entry}; {entry}()"
    os.execv(py, [py, "-c", code, *sys.argv[1:]])


def _wants_help(argv: list[str]) -> bool:
    ros = argv.index("--ros-args") if "--ros-args" in argv else len(argv)
    return any(a in ("-h", "--help") for a in argv[:ros])


def agent() -> None:
    if _wants_help(sys.argv[1:]):
        print(AGENT_USAGE, end="")
        return
    _maybe_reexec("agent")
    from fleetlink.agent import main

    main()


def bridge() -> None:
    if _wants_help(sys.argv[1:]):
        print(BRIDGE_USAGE, end="")
        return
    _maybe_reexec("bridge")
    from fleetlink.relay_ros import main

    main()
