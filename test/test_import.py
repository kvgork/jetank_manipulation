"""Import + pure-logic tests for jetank_manipulation.grasp_server.

These tests load the *source* ``grasp_server.py`` directly by file path and
exercise its pure logic — the SRDF named-target table and the
``_named_target_request`` builder that turns an SRDF state name into a
``moveit_msgs/MotionPlanRequest`` full of ``JointConstraint`` rows.

Heavy / generated deps are stubbed only when absent so the module imports in a
bare environment (no colcon build, no moveit_msgs / control_msgs / generated
GraspObject action). When the real packages are present they are used as-is.
"""

import importlib
import importlib.util
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Stub infrastructure (mirrors jetank_web_control/test/test_labels.py)
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    """Create and register a minimal stub module in sys.modules."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


class _Msg:
    """Generic ROS-message-like stub: any attribute is settable/gettable."""

    def __init__(self):
        # Nested header so `req.workspace_parameters.header.frame_id = ...` works.
        self.header = types.SimpleNamespace(frame_id="")
        self.workspace_parameters = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="")
        )
        self.goal_constraints = []
        self.joint_constraints = []


def _install_stubs():
    """Install stubs for ROS deps not importable in a bare env."""
    # ---- rclpy and its submodules (only the symbols grasp_server imports) ----
    if importlib.util.find_spec("rclpy") is None:
        rclpy_stub = _make_stub("rclpy")
        rclpy_stub.init = lambda *a, **k: None
        rclpy_stub.shutdown = lambda *a, **k: None

        action_stub = _make_stub("rclpy.action")
        for sym in ("ActionClient", "ActionServer", "CancelResponse", "GoalResponse"):
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

    # ---- control_msgs.action.GripperCommand ----
    if importlib.util.find_spec("control_msgs") is None:
        cm = _make_stub("control_msgs")
        cma = _make_stub("control_msgs.action")
        cma.GripperCommand = type("GripperCommand", (), {})
        cm.action = cma

    # ---- moveit_msgs (action + msg) ----
    if importlib.util.find_spec("moveit_msgs") is None:
        mm = _make_stub("moveit_msgs")
        mma = _make_stub("moveit_msgs.action")
        mma.MoveGroup = type("MoveGroup", (), {})
        mm.action = mma

        mmm = _make_stub("moveit_msgs.msg")
        for sym in ("Constraints", "JointConstraint", "MotionPlanRequest",
                    "PlanningOptions"):
            mmm.__dict__[sym] = type(sym, (_Msg,), {})
        # MoveItErrorCodes carries the symbolic int constants the error-name
        # helper reflects over; mirror the ones grasp_server cares about.
        mmm.MoveItErrorCodes = type(
            "MoveItErrorCodes",
            (),
            {"SUCCESS": 1, "FAILURE": 99999, "PLANNING_FAILED": -1,
             "NO_IK_SOLUTION": -31},
        )
        mm.msg = mmm

    # ---- generated GraspObject action (needs a colcon build) ----
    # jetank_manipulation may resolve to the source tree (no generated `action`
    # subpackage) while the install overlay carries the real one — find_spec can
    # then raise ValueError (action.__spec__ is None) instead of returning None.
    # Guard it and only stub when the real generated action is unimportable.
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
        # Reuse the real jetank_manipulation package object if importable so we
        # don't shadow it; otherwise create a bare stub package.
        if "jetank_manipulation" not in sys.modules:
            _make_stub("jetank_manipulation")
        act = _make_stub("jetank_manipulation.action")
        act.GraspObject = type("GraspObject", (), {})
        sys.modules["jetank_manipulation"].action = act


# ---------------------------------------------------------------------------
# Load the SOURCE grasp_server.py by file path (not the installed overlay copy)
# ---------------------------------------------------------------------------

_install_stubs()

_SRC = os.path.join(
    os.path.dirname(__file__), "..", "jetank_manipulation", "grasp_server.py"
)
_SRC = os.path.abspath(_SRC)

try:
    _spec = importlib.util.spec_from_file_location("_grasp_server_under_test", _SRC)
    gs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(gs)
except Exception as exc:  # pragma: no cover - surfaced as a skip
    pytest.skip(f"Could not import grasp_server: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

def test_module_constants():
    assert gs.ARM_GROUP == "arm"
    assert gs.PLANNER_ID == "RRTConnect"
    assert gs._MOVE_ACTION == "/move_action"
    assert gs._GRIPPER_ACTION == "/gripper_controller/gripper_cmd"


# ---------------------------------------------------------------------------
# _SRDF_STATES table
# ---------------------------------------------------------------------------

EXPECTED_STATES = {
    "home": {"S1_joint": 0.0, "S2_joint": 0.0, "S3_joint": 0.0, "S5_joint": 0.0},
    "ready": {"S1_joint": 0.0, "S2_joint": -0.785, "S3_joint": 1.047, "S5_joint": 0.0},
    "grasp_pre": {"S1_joint": 0.0, "S2_joint": 1.222, "S3_joint": -0.262, "S5_joint": 0.0},
    "grasp_reach": {"S1_joint": 0.0, "S2_joint": 1.745, "S3_joint": -0.262, "S5_joint": 0.0},
}


def test_srdf_states_exact_values():
    """Joint targets must match the values mirrored from jetank.srdf."""
    assert set(gs._SRDF_STATES) == set(EXPECTED_STATES)
    for state, joints in EXPECTED_STATES.items():
        assert gs._SRDF_STATES[state] == pytest.approx(joints)


def test_every_state_has_four_dof():
    """All arm group_states cover exactly the 4 actuated joints."""
    for joints in gs._SRDF_STATES.values():
        assert set(joints) == {"S1_joint", "S2_joint", "S3_joint", "S5_joint"}


def test_config_yaml_named_targets_resolve():
    """Every named target referenced by grasp_poses.yaml must exist in the table."""
    cfg = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "grasp_poses.yaml")
    )
    text = open(cfg, encoding="utf-8").read()
    # The yaml lists approach/pre_grasp/grasp/retreat/park under arm_targets.
    for name in ("grasp_pre", "grasp_reach", "home"):
        assert f'"{name}"' in text
        assert name in gs._SRDF_STATES


# ---------------------------------------------------------------------------
# _named_target_request — the SRDF -> MotionPlanRequest builder
# ---------------------------------------------------------------------------

def _build(target):
    return gs._named_target_request(
        gs.ARM_GROUP, target,
        allowed_planning_time=5.0, num_attempts=3,
        vel_scale=0.3, acc_scale=0.3,
    )


def test_request_top_level_fields():
    req = _build("grasp_reach")
    assert req.group_name == "arm"
    assert req.planner_id == "RRTConnect"
    assert req.allowed_planning_time == pytest.approx(5.0)
    assert req.num_planning_attempts == 3
    assert req.max_velocity_scaling_factor == pytest.approx(0.3)
    assert req.max_acceleration_scaling_factor == pytest.approx(0.3)
    assert req.workspace_parameters.header.frame_id == "world"


def test_request_one_constraint_block_named_after_target():
    req = _build("grasp_pre")
    assert len(req.goal_constraints) == 1
    block = req.goal_constraints[0]
    assert block.name == "grasp_pre"


def test_request_joint_constraints_match_srdf():
    """One JointConstraint per DOF, with the exact SRDF position and tolerances."""
    req = _build("grasp_reach")
    jcs = req.goal_constraints[0].joint_constraints
    expected = gs._SRDF_STATES["grasp_reach"]
    assert len(jcs) == len(expected)

    by_name = {jc.joint_name: jc for jc in jcs}
    assert set(by_name) == set(expected)
    for name, val in expected.items():
        jc = by_name[name]
        assert jc.position == pytest.approx(val)
        assert jc.tolerance_above == pytest.approx(0.01)
        assert jc.tolerance_below == pytest.approx(0.01)
        assert jc.weight == pytest.approx(1.0)


def test_scaling_factors_are_passed_through():
    req = gs._named_target_request(
        "arm", "home",
        allowed_planning_time=2.5, num_attempts=7,
        vel_scale=0.5, acc_scale=0.25,
    )
    assert req.allowed_planning_time == pytest.approx(2.5)
    assert req.num_planning_attempts == 7
    assert req.max_velocity_scaling_factor == pytest.approx(0.5)
    assert req.max_acceleration_scaling_factor == pytest.approx(0.25)


def test_unknown_target_raises_value_error():
    with pytest.raises(ValueError):
        _build("definitely_not_a_state")


# ---------------------------------------------------------------------------
# _normalized_quaternion — pure helper (no ROS), always testable
# ---------------------------------------------------------------------------

class _Quat:
    """Minimal duck-typed quaternion for the pure helper."""

    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = x, y, z, w


def test_normalized_quaternion_passes_through_unit():
    # top-down quaternion (1,0,0,0) is already unit -> unchanged, no fallback.
    (x, y, z, w), fellback = gs._normalized_quaternion(_Quat(1.0, 0.0, 0.0, 0.0))
    assert (x, y, z, w) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert fellback is False


def test_normalized_quaternion_normalizes_non_unit():
    import math
    (x, y, z, w), fellback = gs._normalized_quaternion(_Quat(0.0, 0.0, 0.0, 2.0))
    assert fellback is False
    n = math.sqrt(x * x + y * y + z * z + w * w)
    assert n == pytest.approx(1.0)
    assert (x, y, z, w) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_normalized_quaternion_zero_falls_back_to_identity():
    # The catastrophic-failure trigger: an all-zero (unset) quaternion.
    (x, y, z, w), fellback = gs._normalized_quaternion(_Quat(0.0, 0.0, 0.0, 0.0))
    assert fellback is True
    assert (x, y, z, w) == pytest.approx((0.0, 0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# _moveit_error_name — turns the bare MoveItErrorCode int into a symbolic name
# so the TRUE move_group error is visible in logs (not an opaque number).
# ---------------------------------------------------------------------------

def test_moveit_error_name_success():
    assert gs._moveit_error_name(1) == "SUCCESS"


def test_moveit_error_name_catastrophic_failure():
    # 99999 is the genuine move_group "FAILURE" / Catastrophic-failure code.
    assert gs._moveit_error_name(99999) == "FAILURE"


def test_moveit_error_name_unknown_falls_back():
    assert gs._moveit_error_name(-987654) == "UNKNOWN"


# ---------------------------------------------------------------------------
# _pose_target_request — the pose -> MotionPlanRequest builder
#
# Needs the REAL moveit_msgs / geometry_msgs / shape_msgs (a colcon build), so
# it's skipped in a bare env. This is the regression guard for the move_group
# "Catastrophic failure" (error_code 99999) that a malformed pose constraint or
# a non-unit orientation quaternion triggers.
# ---------------------------------------------------------------------------

def _real_pose_msgs_available() -> bool:
    import sys as _sys
    # The bare-env stubs register fake modules in sys.modules; reject those —
    # we only want the genuinely generated messages (they have a real __file__).
    for mod in ("moveit_msgs.msg", "geometry_msgs.msg", "shape_msgs.msg"):
        m = _sys.modules.get(mod)
        if m is None:
            try:
                if importlib.util.find_spec(mod) is None:
                    return False
            except (ValueError, ModuleNotFoundError, ImportError):
                return False
            continue
        # Already imported (possibly a stub): require a filesystem-backed module.
        if getattr(m, "__file__", None) is None:
            return False
    try:
        importlib.import_module("shape_msgs.msg")
        importlib.import_module("geometry_msgs.msg")
        importlib.import_module("moveit_msgs.msg")
    except Exception:
        return False
    return True


_HAVE_POSE_MSGS = _real_pose_msgs_available()

pose_msgs = pytest.mark.skipif(
    not _HAVE_POSE_MSGS,
    reason="real moveit_msgs/geometry_msgs/shape_msgs unavailable (needs colcon build)",
)


def _make_pose_stamped(qx, qy, qz, qw, px=0.209, py=0.0, pz=0.036, frame="base_link"):
    from geometry_msgs.msg import PoseStamped
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x = px
    ps.pose.position.y = py
    ps.pose.position.z = pz
    ps.pose.orientation.x = qx
    ps.pose.orientation.y = qy
    ps.pose.orientation.z = qz
    ps.pose.orientation.w = qw
    return ps


def _build_pose_req(ps, include_orientation=False):
    return gs._pose_target_request(
        gs.ARM_GROUP, ps,
        allowed_planning_time=5.0, num_attempts=3,
        vel_scale=0.3, acc_scale=0.3,
        include_orientation=include_orientation,
    )


@pose_msgs
def test_pose_request_envelope_mirrors_named():
    ps = _make_pose_stamped(1.0, 0.0, 0.0, 0.0)
    req = _build_pose_req(ps)
    assert req.group_name == "arm"
    assert req.planner_id == "RRTConnect"
    assert req.allowed_planning_time == pytest.approx(5.0)
    assert req.num_planning_attempts == 3
    assert req.workspace_parameters.header.frame_id == "base_link"
    assert len(req.goal_constraints) == 1


@pose_msgs
def test_pose_request_position_constraint_populated():
    """constraint_region must carry a BOX primitive AND a matching primitive_pose."""
    ps = _make_pose_stamped(1.0, 0.0, 0.0, 0.0)
    req = _build_pose_req(ps)
    pc = req.goal_constraints[0].position_constraints
    assert len(pc) == 1
    region = pc[0].constraint_region
    assert len(region.primitives) == 1
    assert len(region.primitive_poses) == 1  # one pose per primitive
    from shape_msgs.msg import SolidPrimitive
    assert region.primitives[0].type == SolidPrimitive.BOX
    assert len(region.primitives[0].dimensions) == 3
    assert all(d > 0.0 for d in region.primitives[0].dimensions)
    # The box pose itself must be a valid unit quaternion (w=1).
    assert region.primitive_poses[0].orientation.w == pytest.approx(1.0)
    assert pc[0].link_name == gs.EE_LINK
    assert pc[0].header.frame_id == "base_link"
    assert pc[0].weight > 0.0


@pose_msgs
def test_pose_request_workspace_is_non_degenerate():
    """Workspace box must enclose a real volume (min < max on all axes)."""
    ps = _make_pose_stamped(1.0, 0.0, 0.0, 0.0)
    wp = _build_pose_req(ps).workspace_parameters
    assert wp.min_corner.x < wp.max_corner.x
    assert wp.min_corner.y < wp.max_corner.y
    assert wp.min_corner.z < wp.max_corner.z


@pose_msgs
def test_pose_request_position_only_by_default():
    """Default (pose_use_orientation=False): 1 position_constraint, 0 orientation.

    The 4-DOF arm + KDL IK cannot satisfy a full 6-DOF pose; an orientation
    constraint is the leading move_group "Catastrophic failure" cause, so the
    pose grasp is position-only by default. This is the core regression guard.
    """
    for q in ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)):
        block = _build_pose_req(_make_pose_stamped(*q)).goal_constraints[0]
        assert len(block.position_constraints) == 1
        assert len(block.orientation_constraints) == 0


@pose_msgs
def test_pose_request_with_orientation_has_one_of_each():
    """With pose_use_orientation=True: exactly 1 position + 1 orientation constraint."""
    block = _build_pose_req(
        _make_pose_stamped(1.0, 0.0, 0.0, 0.0), include_orientation=True
    ).goal_constraints[0]
    assert len(block.position_constraints) == 1
    assert len(block.orientation_constraints) == 1


@pose_msgs
def test_pose_request_orientation_is_unit_quaternion_when_enabled():
    import math
    ps = _make_pose_stamped(1.0, 0.0, 0.0, 0.0)
    oc = _build_pose_req(ps, include_orientation=True).goal_constraints[0].orientation_constraints
    assert len(oc) == 1
    o = oc[0].orientation
    n = math.sqrt(o.x ** 2 + o.y ** 2 + o.z ** 2 + o.w ** 2)
    assert n == pytest.approx(1.0)
    assert oc[0].link_name == gs.EE_LINK
    assert oc[0].header.frame_id == "base_link"
    assert oc[0].absolute_x_axis_tolerance > 0.0
    assert oc[0].weight > 0.0


@pose_msgs
def test_pose_request_zero_quaternion_is_repaired_to_unit_when_enabled():
    """The catastrophic-failure case: an all-zero orientation must not survive.

    Only relevant when orientation is explicitly enabled; with it disabled
    (the default) there is no orientation constraint to repair.
    """
    import math
    ps = _make_pose_stamped(0.0, 0.0, 0.0, 0.0)  # unset / degenerate
    oc = _build_pose_req(
        ps, include_orientation=True
    ).goal_constraints[0].orientation_constraints[0]
    o = oc.orientation
    n = math.sqrt(o.x ** 2 + o.y ** 2 + o.z ** 2 + o.w ** 2)
    assert n == pytest.approx(1.0)  # identity, not the zero we passed in
    assert (o.x, o.y, o.z, o.w) == pytest.approx((0.0, 0.0, 0.0, 1.0))
