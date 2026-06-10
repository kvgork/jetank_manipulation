#!/usr/bin/env python3
"""GraspObject action server for the JeTank arm.

Performs an open-loop preset grasp via MoveIt2's /move_action action server.
moveit_py is not available in this RoboStack Humble environment, so we drive
arm motion through the MoveGroup action (moveit_msgs/action/MoveGroup) directly.

Sequence:
  ready -> grasp_pre -> open gripper -> grasp_reach -> close gripper -> ready -> home

Gripper is commanded via control_msgs/action/GripperCommand on
/gripper_controller/gripper_cmd (GripperActionController).
gripper_right_joint is mirrored by the ros2_control native mimic mechanism.

Usage:
  ros2 run jetank_manipulation grasp_server --ros-args -p use_sim_time:=true
  ros2 action send_goal /grasp_object jetank_manipulation/action/GraspObject '{}'
"""

import math
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import GripperCommand
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive

from jetank_manipulation.action import GraspObject

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOVE_ACTION = "/move_action"
_GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

ARM_GROUP = "arm"
PLANNER_ID = "RRTConnect"

# End-effector link of the 'arm' group (chain base_link -> S5_link). Pose-targeted
# grasps constrain THIS link to the requested target pose.
EE_LINK = "S5_link"

# The 4-DOF arm (S1 yaw, S2/S3 pitch, S5 wrist) cannot achieve an arbitrary
# end-effector orientation. We therefore treat the requested orientation as a
# best-effort hint: position is constrained tightly, orientation loosely. These
# generous tolerances (radians) let IK/planning find a solution on the axes the
# arm cannot independently control. ~pi means "effectively unconstrained".
_ORI_TOL_LOOSE = 3.14
# The arm CAN control the approach (top-down) about base yaw fairly well, so we
# leave the loose tolerance on all three axes by default; tighten via a tighter
# value here only if a future, higher-DOF arm warrants it.
_ORI_TOL_TIGHT = 3.14

# Side length of the position-constraint tolerance box around the target point.
_POSITION_BOX_SIZE_M = 0.02

# Half-extent (m) of the workspace_parameters sampling box, centred on the target
# point. move_group samples the position constraint inside this volume; a default
# (zero) WorkspaceParameters box is degenerate and can break IK sampling, so we
# give it a real, generously sized volume around the target.
_WORKSPACE_HALF_EXTENT_M = 1.0


def _moveit_error_name(val: int) -> str:
    """Map a moveit_msgs/MoveItErrorCodes ``.val`` to its symbolic name.

    Reflects the int constants off the generated ``MoveItErrorCodes`` message
    (e.g. ``SUCCESS=1``, ``PLANNING_FAILED=-1``, ``NO_IK_SOLUTION=-31``,
    ``FAILURE=99999`` aka the "Catastrophic failure"). This turns the bare
    integer that move_group returns into a human-readable code so the TRUE error
    is visible in the logs instead of an opaque number. Falls back to
    ``"UNKNOWN"`` for codes not present in the installed message definition.
    """
    try:
        for name, const in vars(MoveItErrorCodes).items():
            if name.isupper() and isinstance(const, int) and const == val:
                return name
    except Exception:  # pragma: no cover - defensive against stub message types
        pass
    return "UNKNOWN"


def _normalized_quaternion(q):
    """Return (x, y, z, w) of *q* normalized to unit length.

    move_group's orientation-constraint validator throws a *catastrophic*
    failure (error_code 99999) on a non-unit — especially an all-zero
    (0,0,0,0) — quaternion, which is the default for an unset
    ``geometry_msgs/Quaternion``. We defensively normalize here and, when the
    input is degenerate (norm ~ 0), fall back to the identity quaternion
    ``(0, 0, 0, 1)`` so the request is never malformed. Returns the components
    plus a bool flag indicating whether a fallback substitution happened.
    """
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1e-6:
        # Degenerate / unset quaternion -> identity (no rotation).
        return (0.0, 0.0, 0.0, 1.0), True
    return (q.x / norm, q.y / norm, q.z / norm, q.w / norm), False


def _named_target_request(
    group_name: str,
    named_target: str,
    allowed_planning_time: float,
    num_attempts: int,
    vel_scale: float,
    acc_scale: float,
) -> MotionPlanRequest:
    """Build a MotionPlanRequest moving *group_name* to a named SRDF target.

    The MoveGroup action goal only accepts explicit joint values or Cartesian
    poses (named-target resolution lives in a separate move_group service, not
    the action). Without moveit_py we therefore embed the known SRDF joint
    values (``_SRDF_STATES``) directly as per-DOF JointConstraints, keeping the
    node self-contained.
    """
    req = MotionPlanRequest()
    req.group_name = group_name
    req.planner_id = PLANNER_ID
    req.allowed_planning_time = allowed_planning_time
    req.num_planning_attempts = num_attempts
    req.max_velocity_scaling_factor = vel_scale
    req.max_acceleration_scaling_factor = acc_scale
    req.workspace_parameters.header.frame_id = "world"

    # Named target joint values from jetank.srdf — kept in sync manually.
    # If the SRDF changes, update _SRDF_STATES below.
    joint_targets = _SRDF_STATES.get(named_target)
    if joint_targets is None:
        raise ValueError(f"Unknown SRDF state: {named_target!r}. Known: {list(_SRDF_STATES)}")

    c = Constraints()
    c.name = named_target
    for jname, jval in joint_targets.items():
        jc = JointConstraint()
        jc.joint_name = jname
        jc.position = jval
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    req.goal_constraints.append(c)
    return req


def _pose_target_request(
    group_name: str,
    pose_stamped: PoseStamped,
    allowed_planning_time: float,
    num_attempts: int,
    vel_scale: float,
    acc_scale: float,
    ee_link: str = EE_LINK,
    position_box_size: float = _POSITION_BOX_SIZE_M,
    include_orientation: bool = False,
    ori_tol_x: float = _ORI_TOL_LOOSE,
    ori_tol_y: float = _ORI_TOL_LOOSE,
    ori_tol_z: float = _ORI_TOL_LOOSE,
    logger=None,
) -> MotionPlanRequest:
    """Build a MotionPlanRequest moving *group_name*'s EE link to a Cartesian pose.

    Modular counterpart to ``_named_target_request`` for the Phase 7 pose-targeted
    grasp path. The goal is a single ``Constraints`` block carrying:

    * a ``PositionConstraint`` on *ee_link*: a small ``SolidPrimitive`` BOX
      (``position_box_size`` per side) centred on ``pose_stamped.pose.position``,
      expressed in ``pose_stamped.header.frame_id`` — **always** included; and
    * (only when ``include_orientation`` is True) an ``OrientationConstraint`` on
      *ee_link* at the (normalized) ``pose_stamped.pose.orientation`` with
      **generous** absolute tolerances.

    Why orientation is OFF by default: the ``arm`` group is **4-DOF**
    (S1/S2/S3/S5) and the SRDF uses the **KDL** IK solver. A 4-DOF arm cannot
    satisfy a full 6-DOF pose (position + orientation); KDL IK throws on the
    over-constrained problem, which surfaces as move_group's *"Catastrophic
    failure"*. Sending position only lets IK/planning pick any wrist orientation
    that reaches the point — which is all a 4-DOF arm can command anyway.
    Orientation is therefore an opt-in best-effort hint (loose tolerances), not
    a rigid target, enabled via ``include_orientation`` for a future, higher-DOF
    arm or deliberate experimentation.

    Robustness: when orientation IS included, the incoming quaternion is
    normalized, and an unset / degenerate (near-zero) quaternion is replaced with
    the identity quaternion. A non-unit (especially all-zero) quaternion is a
    classic trigger for move_group's "Catastrophic failure", since the
    orientation-constraint validator rejects it before any IK is attempted.

    Note this does NOT mutate ``pose_stamped``; it is only read. An optional
    *logger* (an rclpy logger) is used only for a one-line debug dump.
    """
    req = MotionPlanRequest()
    req.group_name = group_name
    req.planner_id = PLANNER_ID
    req.allowed_planning_time = allowed_planning_time
    req.num_planning_attempts = num_attempts
    req.max_velocity_scaling_factor = vel_scale
    req.max_acceleration_scaling_factor = acc_scale

    frame_id = pose_stamped.header.frame_id
    px = pose_stamped.pose.position.x
    py = pose_stamped.pose.position.y
    pz = pose_stamped.pose.position.z

    # Workspace must be a real (non-degenerate) volume: move_group samples the
    # position constraint inside it. The default WorkspaceParameters is a
    # zero-size box (min == max == origin) which can break IK sampling on the
    # pose path. Centre a generous box on the target. (The named-target path
    # never needs this because joint goals don't sample a workspace volume.)
    req.workspace_parameters.header.frame_id = frame_id
    he = _WORKSPACE_HALF_EXTENT_M
    req.workspace_parameters.min_corner.x = px - he
    req.workspace_parameters.min_corner.y = py - he
    req.workspace_parameters.min_corner.z = pz - he
    req.workspace_parameters.max_corner.x = px + he
    req.workspace_parameters.max_corner.y = py + he
    req.workspace_parameters.max_corner.z = pz + he

    c = Constraints()
    c.name = "pose_target"

    # --- Position: tight box around the target point on ee_link ---
    pos_c = PositionConstraint()
    pos_c.header.frame_id = frame_id
    pos_c.link_name = ee_link
    pos_c.weight = 1.0

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [position_box_size, position_box_size, position_box_size]

    bv = BoundingVolume()
    bv.primitives.append(box)
    # The primitive_pose places the box centre at the target position. The
    # constrained point on the link is the link origin offset by
    # target_point_offset (left zero -> the link origin itself). The box pose
    # itself must carry a valid unit quaternion (w=1) or move_group throws.
    primitive_pose = Pose()
    primitive_pose.position.x = px
    primitive_pose.position.y = py
    primitive_pose.position.z = pz
    primitive_pose.orientation.w = 1.0
    bv.primitive_poses.append(primitive_pose)
    pos_c.constraint_region = bv
    c.position_constraints.append(pos_c)

    # --- Orientation: OPT-IN best-effort hint with generous tolerances ---
    # OFF by default: the 4-DOF arm + KDL IK cannot satisfy a full 6-DOF pose,
    # and an orientation constraint on the over-constrained problem is the
    # leading cause of move_group's "Catastrophic failure". When enabled, treat
    # orientation as a loose hint, not a rigid target.
    qx = qy = qz = qw = None
    if include_orientation:
        # Normalize (and identity-fallback) the target quaternion — a non-unit
        # or all-zero quaternion is itself a catastrophic-failure trigger.
        (qx, qy, qz, qw), fellback = _normalized_quaternion(
            pose_stamped.pose.orientation
        )
        if fellback and logger is not None:
            logger.warn(
                "pose_target orientation quaternion was degenerate (near-zero "
                "norm); substituted identity (0,0,0,1)."
            )

        ori_c = OrientationConstraint()
        ori_c.header.frame_id = frame_id
        ori_c.link_name = ee_link
        ori_c.orientation.x = qx
        ori_c.orientation.y = qy
        ori_c.orientation.z = qz
        ori_c.orientation.w = qw
        ori_c.absolute_x_axis_tolerance = ori_tol_x
        ori_c.absolute_y_axis_tolerance = ori_tol_y
        ori_c.absolute_z_axis_tolerance = ori_tol_z
        # Keep the default XYZ_EULER_ANGLES parameterization. ROTATION_VECTOR is
        # not reliably supported across moveit_msgs builds and risks introducing
        # a second catastrophic trigger; with the constraint only ever used with
        # loose tolerances, the default decomposition is acceptable.
        ori_c.weight = 0.5
        c.orientation_constraints.append(ori_c)

    if logger is not None:
        ori_dump = (
            f"({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})"
            if include_orientation
            else "OFF (position-only)"
        )
        logger.debug(
            f"pose_target req: frame='{frame_id}' link='{ee_link}' "
            f"pos=({px:.3f},{py:.3f},{pz:.3f}) "
            f"ori={ori_dump} "
            f"primitives={len(bv.primitives)} primitive_poses={len(bv.primitive_poses)}"
        )

    req.goal_constraints.append(c)
    return req


# SRDF arm group_state joint values (mirror of jetank_moveit_config/config/jetank.srdf).
# Update here if the SRDF changes.
_SRDF_STATES: dict = {
    "home": {
        "S1_joint": 0.0,
        "S2_joint": 0.0,
        "S3_joint": 0.0,
        "S5_joint": 0.0,
    },
    "ready": {
        "S1_joint": 0.0,
        "S2_joint": -0.785,
        "S3_joint": 1.047,
        "S5_joint": 0.0,
    },
    # Forward grasp poses, tuned from RViz (degrees -> rad):
    #   grasp_reach: S2=106deg=1.850, S3=-15deg=-0.262 — reaches floor level.
    #     S2=100deg/1.745 left the gripper ~8-10 cm ABOVE a floor sock (claws shut
    #     in the air above it). S2>=107deg drives the gripper INTO the floor and
    #     the trajectory controller TIMED_OUT / CONTROL_FAILED (jammed target).
    #     106deg is the lowest reach that completes cleanly (swept 110->100 by
    #     1deg). Re-tune here if the floor height changes.
    #   grasp_pre:   derived raised approach (S2=70deg=1.222, same S3)
    "grasp_pre": {
        "S1_joint": 0.0,
        "S2_joint": 1.222,
        "S3_joint": -0.262,
        "S5_joint": 0.0,
    },
    "grasp_reach": {
        "S1_joint": 0.0,
        "S2_joint": 1.8326,
        "S3_joint": -0.262,
        "S5_joint": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Grasp Server Node
# ---------------------------------------------------------------------------


class GraspServer(Node):
    """Action server that executes a deterministic open-loop preset grasp."""

    def __init__(self) -> None:
        super().__init__("grasp_server")

        # Parameters (all tunable via grasp_poses.yaml)
        # Speed-optimised 2026-06-10 (grasp was ~60-90s; see plans/
        # grasp-speed-optimization-plan.md): faster vel/acc, and a tight planning
        # budget because the named targets are JOINT goals that solve instantly —
        # 5s x 3 attempts was the bulk of the per-move "long wait".
        self.declare_parameter("motion.velocity_scaling", 0.6)
        self.declare_parameter("motion.acceleration_scaling", 0.5)
        self.declare_parameter("motion.allowed_planning_time_s", 1.5)
        self.declare_parameter("motion.num_planning_attempts", 1)
        self.declare_parameter("gripper.open_width", 0.04)
        self.declare_parameter("gripper.close_width", 0.0)
        self.declare_parameter("gripper.max_effort", 5.0)
        self.declare_parameter("gripper.dwell_after_open_s", 0.2)
        self.declare_parameter("gripper.dwell_after_close_s", 0.8)
        # approach/retreat empty => SKIP that move. The old "ready" approach swung
        # the arm BACKWARD (S2=-45deg) before grasp_pre (+70deg) for no reason, and
        # retreat-via-ready added a second backward swing. Go home/current ->
        # grasp_pre directly, and retreat straight to home (= park).
        self.declare_parameter("arm_targets.approach", "")
        self.declare_parameter("arm_targets.pre_grasp", "grasp_pre")
        self.declare_parameter("arm_targets.grasp", "grasp_reach")
        self.declare_parameter("arm_targets.retreat", "")
        self.declare_parameter("arm_targets.park", "home")
        # Phase 7 pose-targeted grasp: default pre-grasp/retreat standoff (+Z, m)
        # above the target pose. Used when the goal's approach_height is <= 0.
        self.declare_parameter("pose_grasp.approach_height_m", 0.06)
        # Whether the pose-targeted grasp constrains end-effector ORIENTATION.
        # Default False: the 4-DOF arm + KDL IK cannot satisfy a full 6-DOF pose
        # (position + orientation), and an orientation constraint is the leading
        # cause of move_group's "Catastrophic failure". Position-only lets IK pick
        # any reachable wrist orientation. Set True only for a higher-DOF arm or
        # deliberate experimentation (then it is a loose best-effort hint).
        self.declare_parameter("pose_grasp.pose_use_orientation", False)

        self._cb_group = ReentrantCallbackGroup()

        # MoveGroup action client
        self._move_client = ActionClient(
            self,
            MoveGroup,
            _MOVE_ACTION,
            callback_group=self._cb_group,
        )

        # GripperCommand action client
        # Replaces Float64MultiArray publisher — sends GripperCommand goals to
        # GripperActionController at /gripper_controller/gripper_cmd.
        self._gripper_client = ActionClient(
            self,
            GripperCommand,
            _GRIPPER_ACTION,
            callback_group=self._cb_group,
        )

        # GraspObject action server
        self._action_server = ActionServer(
            self,
            GraspObject,
            "grasp_object",
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"GraspServer started. Waiting for {_MOVE_ACTION} action server..."
        )

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------

    def _goal_cb(self, goal_request):
        self.get_logger().info("GraspObject goal received.")
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().info("GraspObject cancel requested.")
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------
    # Shared MoveGroup plumbing (used by BOTH the preset and pose paths)
    # ------------------------------------------------------------------

    async def _execute_move_request(self, req: MotionPlanRequest, label: str) -> bool:
        """Send a MotionPlanRequest via the MoveGroup action and wait for the result.

        Reused by the legacy named-target path and the Phase 7 pose-target path so
        the goal-send / result-wait / error-code handling lives in exactly one
        place. Returns True iff move_group reports SUCCESS.
        """
        if not self._move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"MoveGroup action server {_MOVE_ACTION} not available."
            )
            return False

        options = PlanningOptions()
        options.plan_only = False
        options.replan = False

        goal_msg = MoveGroup.Goal()
        goal_msg.request = req
        goal_msg.planning_options = options

        self.get_logger().info(f"Sending MoveGroup goal: -> {label}")
        send_goal_future = await self._move_client.send_goal_async(goal_msg)

        if not send_goal_future.accepted:
            self.get_logger().error(f"MoveGroup goal to '{label}' was rejected.")
            return False

        result_future = await send_goal_future.get_result_async()
        motion_result = result_future.result

        # Surface the REAL MoveItErrorCode. moveit_msgs/MoveItErrorCodes: SUCCESS=1.
        # "Catastrophic failure" is the symbolic name for val == 99999, which is a
        # genuine move_group code (not a grasp_server sentinel) and the symptom of
        # KDL IK throwing on an over-constrained 6-DOF goal for the 4-DOF arm.
        error_code = motion_result.error_code.val
        error_name = _moveit_error_name(error_code)
        if error_code == 1:  # MoveItErrorCodes.SUCCESS
            self.get_logger().info(
                f"Reached '{label}' (error_code={error_code} {error_name})."
            )
            return True
        self.get_logger().error(
            f"Motion to '{label}' failed "
            f"(error_code={error_code} {error_name})."
        )
        return False

    async def _execute_cb(self, goal_handle):
        """Run the full grasp sequence asynchronously."""
        self.get_logger().info("Executing GraspObject sequence...")
        feedback_msg = GraspObject.Feedback()
        result = GraspObject.Result()

        # Read parameters
        vel_scale = self.get_parameter("motion.velocity_scaling").value
        acc_scale = self.get_parameter("motion.acceleration_scaling").value
        plan_time = self.get_parameter("motion.allowed_planning_time_s").value
        num_attempts = int(self.get_parameter("motion.num_planning_attempts").value)
        open_width = float(self.get_parameter("gripper.open_width").value)
        close_width = float(self.get_parameter("gripper.close_width").value)
        max_effort = float(self.get_parameter("gripper.max_effort").value)
        dwell_open = float(self.get_parameter("gripper.dwell_after_open_s").value)
        dwell_close = float(self.get_parameter("gripper.dwell_after_close_s").value)

        t_approach = self.get_parameter("arm_targets.approach").value
        t_pre_grasp = self.get_parameter("arm_targets.pre_grasp").value
        t_grasp = self.get_parameter("arm_targets.grasp").value
        t_retreat = self.get_parameter("arm_targets.retreat").value
        t_park = self.get_parameter("arm_targets.park").value

        pose_use_orientation = bool(
            self.get_parameter("pose_grasp.pose_use_orientation").value
        )

        def pub_feedback(stage: str):
            feedback_msg.stage = stage
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"Stage: {stage}")

        async def command_gripper(width: float, effort: float) -> bool:
            """Send a GripperCommand action goal.

            Returns True if the action server accepted the goal and returned a
            result (stall or success); False if the server is absent — the
            caller degrades gracefully and the grasp sequence continues.
            """
            if not self._gripper_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().warn(
                    f"GripperCommand action server {_GRIPPER_ACTION} not available — "
                    "skipping gripper command."
                )
                return False

            gripper_goal = GripperCommand.Goal()
            gripper_goal.command.position = width
            gripper_goal.command.max_effort = effort

            self.get_logger().info(
                f"Sending GripperCommand: position={width:.4f} m, effort={effort:.1f} N"
            )
            send_future = await self._gripper_client.send_goal_async(gripper_goal)

            if not send_future.accepted:
                self.get_logger().error("GripperCommand goal rejected.")
                return False

            result_future = await send_future.get_result_async()
            gripper_result = result_future.result
            self.get_logger().info(
                f"GripperCommand done: position={gripper_result.position:.4f}, "
                f"stalled={gripper_result.stalled}, reached_goal={gripper_result.reached_goal}"
            )
            return True

        async def move_to(target_name: str) -> bool:
            """Plan and execute arm motion to a named SRDF target. Returns True on success."""
            try:
                req = _named_target_request(
                    ARM_GROUP,
                    target_name,
                    plan_time,
                    num_attempts,
                    vel_scale,
                    acc_scale,
                )
            except ValueError as exc:
                self.get_logger().error(str(exc))
                return False
            return await self._execute_move_request(req, target_name)

        async def move_to_pose(pose_stamped: PoseStamped, label: str) -> bool:
            """Plan and execute arm motion so EE_LINK reaches *pose_stamped*.

            Pose-targeted counterpart to ``move_to``; shares the same MoveGroup
            send/wait plumbing via ``_execute_move_request``.
            """
            req = _pose_target_request(
                ARM_GROUP,
                pose_stamped,
                plan_time,
                num_attempts,
                vel_scale,
                acc_scale,
                include_orientation=pose_use_orientation,
                logger=self.get_logger(),
            )
            return await self._execute_move_request(req, label)

        # --- Branch: pose-targeted (Phase 7) vs legacy preset grasp ---
        target_pose = goal_handle.request.target_pose
        if target_pose.header.frame_id:
            self.get_logger().info(
                "GraspObject: pose-targeted grasp "
                f"(frame='{target_pose.header.frame_id}')."
            )
            return await self._run_pose_grasp(
                goal_handle,
                result,
                target_pose,
                approach_height=self._resolve_approach_height(goal_handle),
                t_approach=t_approach,
                t_park=t_park,
                open_width=open_width,
                close_width=close_width,
                max_effort=max_effort,
                dwell_open=dwell_open,
                dwell_close=dwell_close,
                pub_feedback=pub_feedback,
                command_gripper=command_gripper,
                move_to=move_to,
                move_to_pose=move_to_pose,
            )

        self.get_logger().info("GraspObject: legacy preset grasp (no target_pose).")

        # --- Preset grasp sequence (speed-optimised) ---
        # Step 1: move to approach. SKIPPED when t_approach is empty (default) —
        # the old "ready" approach was a redundant backward swing.
        if t_approach:
            pub_feedback("moving_to_approach")
            if not await move_to(t_approach):
                await self._abort_and_retreat(goal_handle, result, move_to, t_park)
                return result

        # Step 2: move to pre-grasp
        pub_feedback("moving_to_pre_grasp")
        if not await move_to(t_pre_grasp):
            await self._abort_and_retreat(goal_handle, result, move_to, t_park)
            return result

        # Step 3: open gripper
        pub_feedback("opening_gripper")
        await command_gripper(open_width, max_effort)
        time.sleep(dwell_open)

        # Step 4: move to grasp
        pub_feedback("moving_to_grasp")
        if not await move_to(t_grasp):
            await self._abort_and_retreat(goal_handle, result, move_to, t_park)
            return result

        # Step 5: close gripper
        pub_feedback("closing_gripper")
        await command_gripper(close_width, max_effort)
        time.sleep(dwell_close)

        # Step 6: retreat. SKIPPED when t_retreat is empty (default) — retreat
        # straight to park (home) instead of swinging through "ready" first.
        if t_retreat:
            pub_feedback("retreating")
            if not await move_to(t_retreat):
                self.get_logger().warn("Retreat failed; attempting park at home.")

        # Step 7: park at home (lifts the grasped sock up)
        pub_feedback("parking")
        await move_to(t_park)

        pub_feedback("done")
        result.success = True
        result.message = "Grasp sequence completed successfully."
        goal_handle.succeed()
        self.get_logger().info("GraspObject: SUCCESS")
        return result

    async def _abort_and_retreat(self, goal_handle, result, move_to_fn, park_target: str):
        """Publish abort stage, retreat to home, and set the goal as aborted."""
        fb = GraspObject.Feedback()
        fb.stage = "aborting_retreat"
        goal_handle.publish_feedback(fb)
        self.get_logger().warn("GraspObject: plan/execute failed — retreating to home.")
        await move_to_fn(park_target)
        result.success = False
        result.message = "Motion failed; retreated to home."
        goal_handle.abort()

    # ------------------------------------------------------------------
    # Phase 7: pose-targeted grasp (modular, separate from the preset path)
    # ------------------------------------------------------------------

    def _resolve_approach_height(self, goal_handle) -> float:
        """Pick the pre-grasp/retreat standoff (m): goal value if >0 else param default."""
        goal_h = float(goal_handle.request.approach_height)
        if goal_h > 0.0:
            return goal_h
        return float(self.get_parameter("pose_grasp.approach_height_m").value)

    @staticmethod
    def _offset_pose_z(pose_stamped: PoseStamped, dz: float) -> PoseStamped:
        """Return a copy of *pose_stamped* raised by *dz* in +Z of its own frame.

        Does not mutate the input. Orientation, frame_id, and stamp are preserved.
        """
        out = PoseStamped()
        out.header.frame_id = pose_stamped.header.frame_id
        out.header.stamp = pose_stamped.header.stamp
        out.pose.position.x = pose_stamped.pose.position.x
        out.pose.position.y = pose_stamped.pose.position.y
        out.pose.position.z = pose_stamped.pose.position.z + dz
        out.pose.orientation = pose_stamped.pose.orientation
        return out

    async def _run_pose_grasp(
        self,
        goal_handle,
        result,
        target_pose: PoseStamped,
        approach_height: float,
        t_approach: str,
        t_park: str,
        open_width: float,
        close_width: float,
        max_effort: float,
        dwell_open: float,
        dwell_close: float,
        pub_feedback,
        command_gripper,
        move_to,
        move_to_pose,
    ):
        """Pose-targeted grasp sequence.

        ready (preset) -> pre_grasp (target + approach_height +Z) -> open gripper
        -> reach (target) -> close gripper -> retreat (back to pre_grasp) -> home.

        Position is constrained tightly; orientation is a best-effort top-down-ish
        hint (loose tolerances) because the arm is 4-DOF. Reuses the same gripper
        and MoveGroup plumbing as the preset path.
        """
        pre_grasp_pose = self._offset_pose_z(target_pose, approach_height)
        self.get_logger().info(
            f"Pose grasp: approach_height={approach_height:.3f} m, "
            f"target z={target_pose.pose.position.z:.3f} m -> "
            f"pre-grasp z={pre_grasp_pose.pose.position.z:.3f} m."
        )

        # Step 1: move to approach (preset 'ready' — known-safe starting pose)
        pub_feedback("moving_to_approach")
        if not await move_to(t_approach):
            await self._abort_and_retreat(goal_handle, result, move_to, t_park)
            return result

        # Step 2: pre-grasp (target pose raised by approach_height)
        pub_feedback("pre_grasp")
        if not await move_to_pose(pre_grasp_pose, "pre_grasp"):
            await self._abort_and_retreat(goal_handle, result, move_to, t_park)
            return result

        # Step 3: open gripper
        pub_feedback("opening_gripper")
        await command_gripper(open_width, max_effort)
        time.sleep(dwell_open)

        # Step 4: reach (the target pose itself)
        pub_feedback("reach")
        if not await move_to_pose(target_pose, "reach"):
            await self._abort_and_retreat(goal_handle, result, move_to, t_park)
            return result

        # Step 5: close gripper
        pub_feedback("close")
        await command_gripper(close_width, max_effort)
        time.sleep(dwell_close)

        # Step 6: retreat (back up to the pre-grasp standoff)
        pub_feedback("retreat")
        if not await move_to_pose(pre_grasp_pose, "retreat"):
            self.get_logger().warn(
                "Pose-grasp retreat failed; attempting park at home."
            )

        # Step 7: park at home (preset)
        pub_feedback("parking")
        await move_to(t_park)

        pub_feedback("done")
        result.success = True
        result.message = "Pose-targeted grasp sequence completed successfully."
        goal_handle.succeed()
        self.get_logger().info("GraspObject (pose): SUCCESS")
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = GraspServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
