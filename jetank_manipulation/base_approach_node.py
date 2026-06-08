#!/usr/bin/env python3
"""Diff-drive base-approach action server for the JeTank.

Hosts the ``/approach_target`` action (``jetank_manipulation/action/ApproachTarget``):
drive the tracked base up to a point until it is ``standoff`` metres ahead, using
a simple proportional rotate-then-drive servo.

Pipeline (per accepted goal):
  - Each control tick, TF the goal ``target`` (a ``geometry_msgs/PointStamped``,
    any frame) into the ``base_frame`` (param, default ``base_link``). If the
    point is already expressed in ``base_frame`` the lookup is the trivial
    identity.
  - From the target in base coords ``(x, y)`` compute ``dist = hypot(x, y)`` and
    ``heading = atan2(y, x)``. Feed those + the standoff/tolerance/gains into the
    pure control law ``approach_control(...)`` which returns ``(lin, ang,
    arrived)``: rotate-to-face first (angular only) until |heading| is small,
    then drive forward until ``dist <= standoff``.
  - Publish the command as a ``geometry_msgs/TwistStamped`` (header stamped with
    the node clock + ``base_frame``) on ``cmd_vel_topic`` (param, default
    ``/diff_drive_controller/cmd_vel``).
  - Publish feedback {distance, heading_error}; succeed (zero Twist) on arrival,
    abort (zero Twist) on timeout or cancel.

The pure control law is factored into the module-level ``approach_control``
function (no ROS types) so it is unit-testable; see ``test/test_base_control.py``.

Usage:
  ros2 run jetank_manipulation base_approach_node --ros-args -p use_sim_time:=true
  ros2 action send_goal /approach_target jetank_manipulation/action/ApproachTarget \
    '{target: {header: {frame_id: base_link}, point: {x: 0.5, y: 0.0, z: 0.0}}, \
      standoff: 0.18, timeout: 20.0}'
"""

import math
from typing import Tuple

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped

import tf2_ros
from tf2_ros import TransformException

# tf2_geometry_msgs registers do_transform for PointStamped on the Buffer.
import tf2_geometry_msgs  # noqa: F401

from jetank_manipulation.action import ApproachTarget


# ---------------------------------------------------------------------------
# Pure control law (no ROS) — unit-testable
# ---------------------------------------------------------------------------


def _clamp(value: float, limit: float) -> float:
    """Symmetric clamp into [-limit, +limit] (``limit`` assumed >= 0)."""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def approach_control(
    dx: float,
    dy: float,
    standoff: float,
    heading_tol: float,
    k_lin: float,
    k_ang: float,
    max_lin: float,
    max_ang: float,
    arrive_tol: float = 0.03,
) -> Tuple[float, float, bool]:
    """Rotate-then-drive diff-drive servo toward a target in the base frame.

    Pure function — no ROS types — so it is trivially unit-testable.

    Parameters
    ----------
    dx, dy:
        Target position in the base frame (metres). ``dx`` is forward, ``dy``
        is left.
    standoff:
        Desired stopping distance: arrive once ``hypot(dx, dy) <= standoff``.
    heading_tol:
        Heading error (rad) below which we stop rotating and start driving
        forward. While |heading| > heading_tol we rotate in place (lin = 0).
    k_lin, k_ang:
        Proportional gains for the linear (distance error) and angular (heading
        error) terms.
    max_lin, max_ang:
        Symmetric saturation limits for the linear / angular commands.

    Returns
    -------
    (lin, ang, arrived)
        ``lin`` forward velocity (m/s), ``ang`` yaw rate (rad/s), and
        ``arrived`` True once within ``standoff`` (caller should stop + succeed).
        On arrival both ``lin`` and ``ang`` are 0.
    """
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)

    # Arrived: within standoff (+ tolerance) -> full stop. The forward term
    # k_lin*(dist-standoff) decays to ~0 as dist approaches standoff, so the base
    # asymptotes to the standoff from above and never crosses `dist <= standoff`
    # exactly — the tolerance band is what actually trips arrival.
    if dist <= standoff + arrive_tol:
        return 0.0, 0.0, True

    # Always servo heading.
    ang = _clamp(k_ang * heading, max_ang)

    # Rotate-to-face first: only drive forward once roughly aligned.
    if abs(heading) > heading_tol:
        lin = 0.0
    else:
        # Drive toward the standoff distance (never command reverse here:
        # dist > standoff is guaranteed above, so the error is positive).
        lin = _clamp(k_lin * (dist - standoff), max_lin)

    return lin, ang, False


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class BaseApproachNode(Node):
    """ApproachTarget action server: proportional diff-drive servo to a point."""

    def __init__(self) -> None:
        super().__init__("base_approach_node")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("cmd_vel_topic", "/diff_drive_controller/cmd_vel")
        self.declare_parameter("k_lin", 0.6)
        self.declare_parameter("k_ang", 1.2)
        self.declare_parameter("max_lin", 0.15)
        self.declare_parameter("max_ang", 0.8)
        self.declare_parameter("heading_tol", 0.15)
        self.declare_parameter("arrive_tol", 0.03)
        self.declare_parameter("control_rate", 10.0)

        self._cb_group = ReentrantCallbackGroup()

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self._cmd_pub = self.create_publisher(TwistStamped, cmd_vel_topic, 10)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self, spin_thread=False
        )

        self._action_server = ActionServer(
            self,
            ApproachTarget,
            "approach_target",
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"base_approach_node ready. Action /approach_target -> cmd_vel '{cmd_vel_topic}'."
        )

    # ------------------------------------------------------------------
    # Action server callbacks
    # ------------------------------------------------------------------

    def _goal_cb(self, goal_request):
        self.get_logger().info("ApproachTarget goal received.")
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().info("ApproachTarget cancel requested.")
        return CancelResponse.ACCEPT

    async def _execute_cb(self, goal_handle):
        goal = goal_handle.request
        base_frame = self.get_parameter("base_frame").value
        k_lin = float(self.get_parameter("k_lin").value)
        k_ang = float(self.get_parameter("k_ang").value)
        max_lin = float(self.get_parameter("max_lin").value)
        max_ang = float(self.get_parameter("max_ang").value)
        heading_tol = float(self.get_parameter("heading_tol").value)
        arrive_tol = float(self.get_parameter("arrive_tol").value)
        control_rate = float(self.get_parameter("control_rate").value)

        standoff = float(goal.standoff)
        timeout = float(goal.timeout)
        period = 1.0 / control_rate if control_rate > 0.0 else 0.1

        result = ApproachTarget.Result()

        self.get_logger().info(
            f"ApproachTarget: target frame='{goal.target.header.frame_id}' "
            f"standoff={standoff:.3f} timeout={timeout:.1f}s -> base_frame='{base_frame}'."
        )

        start_ns = self.get_clock().now().nanoseconds
        deadline_ns = start_ns + int(timeout * 1e9) if timeout > 0.0 else None

        last_dist = float("nan")
        rate = self.create_rate(control_rate if control_rate > 0.0 else 10.0)

        try:
            while rclpy.ok():
                # Cancellation.
                if goal_handle.is_cancel_requested:
                    self._stop()
                    goal_handle.canceled()
                    result.success = False
                    result.final_distance = last_dist
                    result.message = "ApproachTarget canceled."
                    self.get_logger().info(result.message)
                    return result

                # Timeout.
                if deadline_ns is not None and (
                    self.get_clock().now().nanoseconds > deadline_ns
                ):
                    self._stop()
                    goal_handle.abort()
                    result.success = False
                    result.final_distance = last_dist
                    result.message = (
                        f"ApproachTarget timed out after {timeout:.1f}s "
                        f"(last distance={last_dist:.3f}m)."
                    )
                    self.get_logger().warn(result.message)
                    return result

                # TF the target into the base frame for this tick.
                dx_dy = self._target_in_base(goal.target, base_frame)
                if dx_dy is None:
                    # Transient TF gap: hold still, retry next tick.
                    self._stop()
                    self._sleep_rate(rate, period)
                    continue

                dx, dy = dx_dy
                lin, ang, arrived = approach_control(
                    dx, dy, standoff, heading_tol,
                    k_lin, k_ang, max_lin, max_ang,
                    arrive_tol=arrive_tol,
                )
                last_dist = math.hypot(dx, dy)
                heading = math.atan2(dy, dx)

                fb = ApproachTarget.Feedback()
                fb.distance = float(last_dist)
                fb.heading_error = float(heading)
                goal_handle.publish_feedback(fb)

                if arrived:
                    self._stop()
                    goal_handle.succeed()
                    result.success = True
                    result.final_distance = float(last_dist)
                    result.message = (
                        f"Arrived within standoff: distance={last_dist:.3f}m "
                        f"<= standoff={standoff:.3f}m."
                    )
                    self.get_logger().info(result.message)
                    return result

                self._publish_cmd(lin, ang, base_frame)
                self._sleep_rate(rate, period)

        finally:
            # Defensive: never leave the base creeping if we exit unexpectedly.
            self._stop()

        # rclpy shutdown mid-goal.
        self._stop()
        result.success = False
        result.final_distance = last_dist
        result.message = "ApproachTarget aborted: node shutting down."
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _target_in_base(self, target, base_frame):
        """Return (dx, dy) of *target* in *base_frame*, or None on TF failure."""
        if target.header.frame_id == base_frame or not target.header.frame_id:
            return float(target.point.x), float(target.point.y)
        try:
            # Latest available transform (Time() == 0).
            transformed = self._tf_buffer.transform(
                target,
                base_frame,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            return float(transformed.point.x), float(transformed.point.y)
        except TransformException as exc:
            self.get_logger().warn(
                f"TF '{target.header.frame_id}' -> '{base_frame}' failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def _publish_cmd(self, lin: float, ang: float, base_frame: str) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = float(lin)
        msg.twist.angular.z = float(ang)
        self._cmd_pub.publish(msg)

    def _stop(self) -> None:
        base_frame = self.get_parameter("base_frame").value
        self._publish_cmd(0.0, 0.0, base_frame)

    def _sleep_rate(self, rate, period: float) -> None:
        """Sleep one control period.

        Uses a plain time.sleep so the loop ticks even without a wall executor
        thread driving Rate (Rate.sleep can deadlock on a single-thread spin);
        the MultiThreadedExecutor keeps servicing TF / cancel concurrently.
        """
        import time

        time.sleep(period)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = BaseApproachNode()
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
