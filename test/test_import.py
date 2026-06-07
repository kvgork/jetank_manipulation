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
        mm.msg = mmm

    # ---- generated GraspObject action (needs a colcon build) ----
    if importlib.util.find_spec("jetank_manipulation.action") is None:
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
