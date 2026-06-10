"""Unit tests for the pure diff-drive control law in ``base_approach_node``.

The node module imports rclpy / tf2 / ROS message packages at import time. To
keep the *pure* control law (``approach_control``) testable in a bare
environment, we stub the absent ROS deps before loading the source module by
file path — mirroring ``test_grasp_pose.py``. When the real packages are present
(post colcon build) they are used as-is.
"""

import importlib
import importlib.util
import math
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Stub the ROS-only imports so the module loads with no ROS available.
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_stubs():
    if importlib.util.find_spec("rclpy") is None:
        rclpy_stub = _make_stub("rclpy")
        rclpy_stub.ok = lambda *a, **k: True
        rclpy_stub.init = lambda *a, **k: None
        rclpy_stub.shutdown = lambda *a, **k: None

        dur_stub = _make_stub("rclpy.duration")
        dur_stub.Duration = type("Duration", (), {})
        rclpy_stub.duration = dur_stub

        action_stub = _make_stub("rclpy.action")
        for sym in ("ActionServer", "CancelResponse", "GoalResponse"):
            setattr(action_stub, sym, type(sym, (), {}))
        rclpy_stub.action = action_stub

        cbg_stub = _make_stub("rclpy.callback_groups")
        cbg_stub.ReentrantCallbackGroup = type("ReentrantCallbackGroup", (), {})
        rclpy_stub.callback_groups = cbg_stub

        ex_stub = _make_stub("rclpy.executors")
        ex_stub.MultiThreadedExecutor = type("MultiThreadedExecutor", (), {})
        rclpy_stub.executors = ex_stub

        node_stub = _make_stub("rclpy.node")
        node_stub.Node = object
        rclpy_stub.node = node_stub

    if importlib.util.find_spec("geometry_msgs") is None:
        gm = _make_stub("geometry_msgs")
        gmm = _make_stub("geometry_msgs.msg")
        gmm.TwistStamped = type("TwistStamped", (), {})
        gm.msg = gmm

    if importlib.util.find_spec("tf2_ros") is None:
        tf2 = _make_stub("tf2_ros")
        tf2.Buffer = type("Buffer", (), {})
        tf2.TransformListener = type("TransformListener", (), {})
        tf2.TransformException = type("TransformException", (Exception,), {})

    if importlib.util.find_spec("tf2_geometry_msgs") is None:
        _make_stub("tf2_geometry_msgs")

    # jetank_manipulation may resolve to the source tree (no generated `action`
    # subpackage) while the install overlay carries the real one. find_spec can
    # then raise ValueError (action.__spec__ is None) instead of returning None,
    # so guard it and only stub when the real generated action is unimportable.
    try:
        _have_action = importlib.util.find_spec("jetank_manipulation.action") is not None
    except (ValueError, ModuleNotFoundError, ImportError):
        _have_action = False
    if _have_action:
        try:
            importlib.import_module("jetank_manipulation.action")
        except Exception:
            _have_action = False
    if not _have_action:
        jm = sys.modules.get("jetank_manipulation") or _make_stub(
            "jetank_manipulation"
        )
        jma = _make_stub("jetank_manipulation.action")
        jma.ApproachTarget = type("ApproachTarget", (), {})
        jm.action = jma


_install_stubs()

_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "jetank_manipulation",
        "base_approach_node.py",
    )
)

try:
    _spec = importlib.util.spec_from_file_location(
        "_base_approach_under_test", _SRC
    )
    ba = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ba)
except Exception as exc:  # pragma: no cover - surfaced as a skip
    pytest.skip(f"Could not import base_approach_node: {exc}", allow_module_level=True)


# Default gains matching the node param defaults.
_GAINS = dict(k_lin=0.6, k_ang=1.2, max_lin=0.15, max_ang=0.8)
_HTOL = 0.15


def _ctrl(dx, dy, standoff=0.18, heading_tol=_HTOL):
    return ba.approach_control(dx, dy, standoff, heading_tol, **_GAINS)


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------

def test_clamp_symmetric():
    assert ba._clamp(5.0, 2.0) == 2.0
    assert ba._clamp(-5.0, 2.0) == -2.0
    assert ba._clamp(1.0, 2.0) == 1.0
    assert ba._clamp(0.0, 2.0) == 0.0


# ---------------------------------------------------------------------------
# Arrival
# ---------------------------------------------------------------------------

def test_arrived_within_standoff_stops():
    lin, ang, arrived = _ctrl(0.1, 0.0, standoff=0.18)
    assert arrived is True
    assert lin == 0.0
    assert ang == 0.0


def test_arrived_exactly_at_standoff():
    lin, ang, arrived = _ctrl(0.18, 0.0, standoff=0.18)
    assert arrived is True
    assert lin == 0.0 and ang == 0.0


# ---------------------------------------------------------------------------
# Rotate-to-face first
# ---------------------------------------------------------------------------

def test_rotates_in_place_when_misaligned():
    # Target straight to the left (heading = +pi/2): big heading error.
    lin, ang, arrived = _ctrl(0.0, 1.0, standoff=0.18)
    assert arrived is False
    assert lin == 0.0  # no forward drive while rotating to face
    assert ang > 0.0   # turn left (positive yaw rate) toward the target


def test_rotates_right_for_target_on_right():
    lin, ang, arrived = _ctrl(0.0, -1.0, standoff=0.18)
    assert arrived is False
    assert lin == 0.0
    assert ang < 0.0   # turn right


def test_angular_clamped_to_max():
    # Target directly behind (heading ~ pi): k_ang*pi >> max_ang.
    lin, ang, arrived = _ctrl(-1.0, 1e-9, standoff=0.18)
    assert abs(ang) == pytest.approx(_GAINS["max_ang"])


# ---------------------------------------------------------------------------
# Drive forward once aligned
# ---------------------------------------------------------------------------

def test_drives_forward_when_aligned():
    # Straight ahead, well beyond standoff.
    lin, ang, arrived = _ctrl(1.0, 0.0, standoff=0.18)
    assert arrived is False
    assert lin > 0.0
    assert ang == pytest.approx(0.0)


def test_linear_clamped_to_max():
    # Far away straight ahead -> k_lin*(dist-standoff) >> max_lin.
    lin, _, _ = _ctrl(5.0, 0.0, standoff=0.18)
    assert lin == pytest.approx(_GAINS["max_lin"])


def test_linear_is_proportional_to_distance_error():
    # Just past the heading tolerance threshold, small distance error so the
    # proportional term stays below saturation.
    dx = 0.18 + 0.05  # 0.05 m past standoff
    lin, ang, arrived = _ctrl(dx, 0.0, standoff=0.18)
    assert lin == pytest.approx(_GAINS["k_lin"] * 0.05)
    assert ang == pytest.approx(0.0)


def test_forward_drive_only_within_heading_tol():
    # Heading error just inside tol -> may drive; just outside -> must not.
    standoff = 0.18
    dist = 1.0
    inside = ba.approach_control(
        dist * math.cos(0.10), dist * math.sin(0.10),
        standoff, _HTOL, **_GAINS,
    )
    outside = ba.approach_control(
        dist * math.cos(0.30), dist * math.sin(0.30),
        standoff, _HTOL, **_GAINS,
    )
    assert inside[0] > 0.0       # within tol -> drives
    assert outside[0] == 0.0     # beyond tol -> rotate only
