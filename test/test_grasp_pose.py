"""Unit tests for the pure grasp-pose math in ``grasp_pose_node``.

The node module imports rclpy / ROS message packages at import time. To keep the
*pure* math (``compute_grasp_fields``, ``pca_long_axis_yaw``,
``top_down_quaternion``) testable in a bare environment (only numpy required),
we stub the absent ROS deps before loading the source module by file path —
mirroring the stub approach in ``test_import.py``. When the real packages are
present (post colcon build) they are used as-is.
"""

import importlib
import importlib.util
import math
import os
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub the ROS-only imports so the module loads with just numpy available.
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

        action_stub = _make_stub("rclpy.action")
        action_stub.ActionClient = type("ActionClient", (), {})
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

        qos_stub = _make_stub("rclpy.qos")
        for sym in (
            "DurabilityPolicy",
            "HistoryPolicy",
            "QoSProfile",
            "ReliabilityPolicy",
        ):
            setattr(qos_stub, sym, type(sym, (), {}))
        rclpy_stub.qos = qos_stub

    if importlib.util.find_spec("std_srvs") is None:
        ss = _make_stub("std_srvs")
        ssr = _make_stub("std_srvs.srv")
        ssr.Trigger = type("Trigger", (), {})
        ss.srv = ssr

    if importlib.util.find_spec("geometry_msgs") is None:
        gm = _make_stub("geometry_msgs")
        gmm = _make_stub("geometry_msgs.msg")
        for sym in ("PoseStamped", "Quaternion"):
            setattr(gmm, sym, type(sym, (), {}))
        gm.msg = gmm

    if importlib.util.find_spec("visualization_msgs") is None:
        vm = _make_stub("visualization_msgs")
        vmm = _make_stub("visualization_msgs.msg")
        vmm.Marker = type("Marker", (), {})
        vm.msg = vmm

    if importlib.util.find_spec("sensor_msgs_py") is None:
        smp = _make_stub("sensor_msgs_py")
        pc2 = _make_stub("sensor_msgs_py.point_cloud2")
        pc2.read_points = lambda *a, **k: np.empty((0,))
        smp.point_cloud2 = pc2

    if importlib.util.find_spec("jetank_detection") is None:
        jd = _make_stub("jetank_detection")
        jda = _make_stub("jetank_detection.action")
        jda.SegmentSocks = type("SegmentSocks", (), {})
        jd.action = jda


_install_stubs()

_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "jetank_manipulation", "grasp_pose_node.py"
    )
)

try:
    _spec = importlib.util.spec_from_file_location("_grasp_pose_under_test", _SRC)
    gp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(gp)
except Exception as exc:  # pragma: no cover - surfaced as a skip
    pytest.skip(f"Could not import grasp_pose_node: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# wrap_to_half_pi
# ---------------------------------------------------------------------------

def test_wrap_to_half_pi_range():
    for a in np.linspace(-10.0, 10.0, 201):
        w = gp.wrap_to_half_pi(float(a))
        assert -math.pi / 2.0 < w <= math.pi / 2.0 + 1e-12


def test_wrap_to_half_pi_symmetry():
    # yaw and yaw+pi command the same symmetric gripper opening.
    for a in (0.1, 0.7, -0.4, 1.2):
        assert gp.wrap_to_half_pi(a) == pytest.approx(gp.wrap_to_half_pi(a + math.pi))


# ---------------------------------------------------------------------------
# pca_long_axis_yaw
# ---------------------------------------------------------------------------

def test_pca_long_axis_along_x():
    # Cloud elongated along +x -> principal axis angle ~ 0 (or pi, same line).
    pts = np.column_stack((np.linspace(-1, 1, 50), np.zeros(50)))
    theta = gp.pca_long_axis_yaw(pts)
    assert theta is not None
    assert math.sin(theta) == pytest.approx(0.0, abs=1e-6)


def test_pca_long_axis_along_y():
    pts = np.column_stack((np.zeros(50), np.linspace(-1, 1, 50)))
    theta = gp.pca_long_axis_yaw(pts)
    assert theta is not None
    # axis is vertical -> |cos(theta)| ~ 0
    assert math.cos(theta) == pytest.approx(0.0, abs=1e-6)


def test_pca_degenerate_returns_none():
    assert gp.pca_long_axis_yaw(np.zeros((5, 2))) is None
    assert gp.pca_long_axis_yaw(np.zeros((1, 2))) is None
    assert gp.pca_long_axis_yaw(np.empty((0, 2))) is None


# ---------------------------------------------------------------------------
# compute_grasp_fields
# ---------------------------------------------------------------------------

def _elongated_x_cloud():
    """A flat sock elongated along +x, top at z=0.20."""
    n = 60
    xs = np.linspace(-0.10, 0.10, n)
    ys = np.full(n, 0.0)
    zs = np.full(n, 0.20)
    return np.column_stack((xs, ys, zs))


def test_elongated_x_cloud_yields_perpendicular_yaw():
    pts = _elongated_x_cloud()
    (x, y, z), yaw = gp.compute_grasp_fields(
        pts, centroid=(0.5, 0.1, 0.2), dimensions=(0.2, 0.02, 0.02),
        yaw_from_pca=True,
    )
    # Long axis along x (theta=0) -> grasp yaw = theta + pi/2 = pi/2.
    assert abs(yaw) == pytest.approx(math.pi / 2.0, abs=1e-6)


def test_centroid_xy_is_used_for_position():
    pts = _elongated_x_cloud()
    (x, y, z), yaw = gp.compute_grasp_fields(
        pts, centroid=(0.5, 0.1, 0.05), dimensions=(0.2, 0.02, 0.02),
    )
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.1)


def test_top_z_plus_offset():
    pts = _elongated_x_cloud()  # max z = 0.20
    (_, _, z), _ = gp.compute_grasp_fields(
        pts, centroid=(0.5, 0.1, 0.05), dimensions=(0.2, 0.02, 0.02),
        approach_z_offset=0.03,
    )
    assert z == pytest.approx(0.23)


def test_yaw_fixed_when_pca_disabled():
    pts = _elongated_x_cloud()
    (_, _, _), yaw = gp.compute_grasp_fields(
        pts, centroid=(0.5, 0.1, 0.2), dimensions=(0.2, 0.02, 0.02),
        yaw_from_pca=False, default_yaw=0.25,
    )
    assert yaw == pytest.approx(0.25)


def test_empty_cloud_falls_back_to_centroid_z_and_default_yaw():
    empty = np.empty((0, 3))
    (x, y, z), yaw = gp.compute_grasp_fields(
        empty, centroid=(0.4, -0.2, 0.07), dimensions=(0.0, 0.0, 0.0),
        yaw_from_pca=True, approach_z_offset=0.0, default_yaw=0.0,
    )
    assert (x, y) == pytest.approx((0.4, -0.2))
    assert z == pytest.approx(0.07)  # no points -> centroid z
    assert yaw == pytest.approx(0.0)  # PCA unusable -> default


# ---------------------------------------------------------------------------
# top_down_quaternion
# ---------------------------------------------------------------------------

def test_quaternion_is_unit():
    for yaw in (0.0, 0.5, -1.2, math.pi / 2.0):
        q = gp.top_down_quaternion(yaw)
        n = math.sqrt(sum(c * c for c in q))
        assert n == pytest.approx(1.0, abs=1e-9)


def test_quaternion_zero_yaw_is_roll_pi():
    # yaw=0 -> pure roll of pi about X -> (x, y, z, w) = (1, 0, 0, 0).
    qx, qy, qz, qw = gp.top_down_quaternion(0.0)
    assert qx == pytest.approx(1.0, abs=1e-9)
    assert qy == pytest.approx(0.0, abs=1e-9)
    assert qz == pytest.approx(0.0, abs=1e-9)
    assert qw == pytest.approx(0.0, abs=1e-9)


def test_quaternion_points_approach_axis_down():
    # The tool +Z axis, rotated by the grasp quaternion, must point along -Z
    # of the target frame (top-down approach) for any yaw.
    for yaw in (0.0, 0.6, -1.1, math.pi / 3.0):
        qx, qy, qz, qw = gp.top_down_quaternion(yaw)
        # Rotation matrix third column = R * [0,0,1] (image of tool +Z).
        zx = 2.0 * (qx * qz + qw * qy)
        zy = 2.0 * (qy * qz - qw * qx)
        zz = 1.0 - 2.0 * (qx * qx + qy * qy)
        assert zx == pytest.approx(0.0, abs=1e-9)
        assert zy == pytest.approx(0.0, abs=1e-9)
        assert zz == pytest.approx(-1.0, abs=1e-9)
