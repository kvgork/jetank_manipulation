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

import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import GripperCommand
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)

from jetank_manipulation.action import GraspObject

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOVE_ACTION = "/move_action"
_GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

ARM_GROUP = "arm"
PLANNER_ID = "RRTConnect"


def _named_target_request(
    group_name: str,
    named_target: str,
    allowed_planning_time: float,
    num_attempts: int,
    vel_scale: float,
    acc_scale: float,
) -> MotionPlanRequest:
    """Build a MotionPlanRequest that moves *group_name* to *named_target*.

    MoveIt2 resolves named targets from SRDF group_states at the move_group
    side, so the client just needs to populate goal_constraints with a single
    JointConstraint per DOF using the name from the SRDF.  However, the simpler
    approach that move_group recognises is to set workspace_parameters.header
    and embed the named target as a GoalConstraint; but MoveGroup action does
    not expose a «named target» field directly.  We therefore use the
    moveit_msgs MotionPlanRequest with the goal expressed as joint constraints
    built from the known SRDF values, OR we use the «named_target» workaround
    that move_group accepts via the pipeline_id field naming a shorthand goal.

    The most robust approach without moveit_py is to set
    ``workspace_parameters.header.frame_id = "world"`` and specify
    ``goal_constraints`` as empty list but fill ``goal_constraints`` with a
    single Constraints message whose ``name`` equals the SRDF state name.
    MoveIt's OMPL pipeline reads named constraints from the SRDF when the
    Constraints message has an empty joint_constraints list but a non-empty name
    matching a group_state.

    Actually, the clean path for named targets is: move_group's SetNamedJointTarget
    is a ROS service, not part of the action goal. The action goal only accepts
    explicit joint values or Cartesian poses. So we embed the known SRDF joint
    values directly in the request. This keeps the node self-contained.
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
    # Tuned 2026-06-06 against sim (widened S3 +/-2.356) to reach as low in front
    # as the arm allows: pre ~0.19 m, reach ~0.15 m above floor. The arm cannot
    # touch the floor with the current URDF primitive link lengths — raise the
    # target onto a low riser, or correct link lengths to real dims.
    # LOWEST COLLISION-FREE forward reach (2026-06-06). The arm can kinematically
    # reach ~0.06 m above the floor, but below ~0.10 m the wrist (S5_link) self-
    # collides with the arm-mounted camera (camera_link) and move_group refuses
    # to plan there. Lowest move_group-valid pose:
    #   grasp_pre  S2=1.0 S3=0.0 -> x~0.225 z~0.152 (forward, raised)
    #   grasp_reach S2=1.0 S3=1.0 -> x~0.147 z~0.105 (forward, ~0.08 m above floor)
    # To go lower: resolve the S5_link<->camera_link collision (relocate camera or
    # disable the pair if the primitive boxes are over-conservative).
    "grasp_pre": {
        "S1_joint": 0.0,
        "S2_joint": 1.0,
        "S3_joint": 0.0,
        "S5_joint": 0.0,
    },
    "grasp_reach": {
        "S1_joint": 0.0,
        "S2_joint": 1.0,
        "S3_joint": 1.0,
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
        self.declare_parameter("motion.velocity_scaling", 0.3)
        self.declare_parameter("motion.acceleration_scaling", 0.3)
        self.declare_parameter("motion.allowed_planning_time_s", 5.0)
        self.declare_parameter("motion.num_planning_attempts", 3)
        self.declare_parameter("gripper.open_width", 0.04)
        self.declare_parameter("gripper.close_width", 0.0)
        self.declare_parameter("gripper.max_effort", 5.0)
        self.declare_parameter("gripper.dwell_after_open_s", 0.5)
        self.declare_parameter("gripper.dwell_after_close_s", 0.8)
        self.declare_parameter("arm_targets.approach", "ready")
        self.declare_parameter("arm_targets.pre_grasp", "grasp_pre")
        self.declare_parameter("arm_targets.grasp", "grasp_reach")
        self.declare_parameter("arm_targets.retreat", "ready")
        self.declare_parameter("arm_targets.park", "home")

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

        def pub_feedback(stage: str):
            feedback_msg.stage = stage
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"Stage: {stage}")

        async def command_gripper(width: float, effort: float) -> bool:
            """Send a GripperCommand action goal. Returns True if the action
            server accepted the goal and returned a result (stall or success).
            Returns False if the action server is absent — caller degrades
            gracefully and the grasp sequence continues.
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
            if not self._move_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error(
                    f"MoveGroup action server {_MOVE_ACTION} not available."
                )
                return False
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

            options = PlanningOptions()
            options.plan_only = False
            options.replan = False

            goal_msg = MoveGroup.Goal()
            goal_msg.request = req
            goal_msg.planning_options = options

            self.get_logger().info(f"Sending MoveGroup goal: -> {target_name}")
            send_goal_future = await self._move_client.send_goal_async(goal_msg)

            if not send_goal_future.accepted:
                self.get_logger().error(f"MoveGroup goal to '{target_name}' was rejected.")
                return False

            result_future = await send_goal_future.get_result_async()
            motion_result = result_future.result

            # moveit_msgs/MoveItErrorCodes: SUCCESS = 1
            error_code = motion_result.error_code.val
            if error_code == 1:
                self.get_logger().info(f"Reached '{target_name}' (error_code=SUCCESS).")
                return True
            else:
                self.get_logger().error(
                    f"Motion to '{target_name}' failed (error_code={error_code})."
                )
                return False

        # --- Grasp sequence ---
        # Step 1: move to approach (ready)
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

        # Step 6: retreat to ready
        pub_feedback("retreating")
        if not await move_to(t_retreat):
            # Already attempted grasp; retreat failure is non-fatal but logged
            self.get_logger().warn("Retreat to ready failed; attempting park at home.")

        # Step 7: park at home
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
