#!/usr/bin/env python3
"""Mobile-manipulation grasp coordinator for the JeTank (Phase 7).

A *thin* state machine that ORCHESTRATES the existing modular pieces purely via
their ROS interfaces — it owns no perception, motion or driving logic of its own.
Triggered by ``~/execute_sock_grasp`` (``std_srvs/srv/Trigger``); on trigger it
runs:

  1. SEGMENT      ActionClient /segment_socks (jetank_detection/SegmentSocks)
                  with target_frame = ``target_frame`` (param). No sock -> fail.
                  On success, build the top-down grasp PoseStamped in
                  ``target_frame`` (base_link) and transform it into the
                  world-fixed ``world_frame`` (param, default ``odom``); the
                  stored odom-frame pose is the grasp target captured while the
                  sock is still in view.
  2. REACH_CHECK  Is the sock centroid within the arm's reachable envelope
                  (``arm_reach`` about ``arm_base_xy``)? If yes, skip APPROACH.
  3. APPROACH     ActionClient /approach_target (jetank_manipulation/ApproachTarget)
                  with target = the sock centroid, standoff = ``approach_standoff``.
                  (only when not already reachable.)
  4. GRASP        ActionClient /grasp_object (jetank_manipulation/GraspObject):
                  transform the stored world-frame grasp pose back into
                  ``target_frame`` at the latest TF (the base has driven up, so
                  the sock is no longer visible to the fixed camera — we grasp
                  open-loop from the remembered odom pose) and send it as the
                  grasp target_pose.
  5. DONE / FAILED  Report via the Trigger response.

Any missing server / timeout / rejection / TF lookup failure at any step yields
a *graceful* ``Trigger`` response with ``success=false`` and a clear message —
the coordinator never raises into the service.

Uses ActionClients on a ReentrantCallbackGroup under a MultiThreadedExecutor (the
same idiom as ``grasp_pose_node.py``): each sub-call sends the goal then spins the
executor until the future resolves or a per-step timeout elapses.

Usage:
  ros2 run jetank_manipulation mobile_grasp_coordinator --ros-args -p use_sim_time:=true
  ros2 service call /mobile_grasp_coordinator/execute_sock_grasp std_srvs/srv/Trigger '{}'
"""

import math

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

import tf2_ros
# Importing tf2_geometry_msgs registers the PoseStamped transform so that
# tf2_ros.Buffer.transform() can convert PoseStamped between frames.
import tf2_geometry_msgs  # noqa: F401

from std_srvs.srv import Trigger
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion

from jetank_detection.action import SegmentSocks
from jetank_manipulation.action import ApproachTarget, GraspObject

# Reuse the top-down grasp orientation math from grasp_pose_node (DRY within
# the package). top_down_quaternion(yaw) -> (x, y, z, w) with the tool +Z
# pointing down (roll = pi) and yaw spun about the vertical approach axis.
from jetank_manipulation.grasp_pose_node import top_down_quaternion


class MobileGraspCoordinator(Node):
    """Trigger -> SEGMENT -> REACH_CHECK -> [APPROACH] -> GRASP -> DONE."""

    def __init__(self) -> None:
        super().__init__("mobile_grasp_coordinator")

        self.declare_parameter("min_score", 0.3)
        self.declare_parameter("max_range", 3.0)
        self.declare_parameter("arm_reach", 0.22)
        self.declare_parameter("arm_base_xy", [0.06, 0.0])
        self.declare_parameter("approach_standoff", 0.18)
        # 'preset' (default): grasp_server runs its tuned grasp_reach (validated for
        # the 4-DOF arm; a Cartesian floor pose is infeasible — camera self-collision).
        # 'pose': send the remembered world grasp pose (for elevated/reachable targets).
        self.declare_parameter("grasp_mode", "preset")
        self.declare_parameter("target_frame", "base_link")
        # World-fixed frame used to remember the grasp pose across the drive.
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("segment_action", "/segment_socks")
        self.declare_parameter("approach_action", "/approach_target")
        self.declare_parameter("grasp_action", "/grasp_object")
        # Per-step timeouts (s).
        self.declare_parameter("segment_timeout_s", 10.0)
        self.declare_parameter("approach_timeout_s", 30.0)
        self.declare_parameter("grasp_timeout_s", 60.0)

        # TF: capture the grasp pose in world_frame while the sock is visible,
        # then retrieve it back into target_frame after the base has driven up.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # World-fixed grasp pose captured during SEGMENT (PoseStamped in
        # world_frame), retrieved into target_frame at GRASP time.
        self._grasp_pose_world = None

        self._cb_group = ReentrantCallbackGroup()

        self._seg_client = ActionClient(
            self, SegmentSocks,
            self.get_parameter("segment_action").value,
            callback_group=self._cb_group,
        )
        self._approach_client = ActionClient(
            self, ApproachTarget,
            self.get_parameter("approach_action").value,
            callback_group=self._cb_group,
        )
        self._grasp_client = ActionClient(
            self, GraspObject,
            self.get_parameter("grasp_action").value,
            callback_group=self._cb_group,
        )

        # Latched grasp pose for RViz (optional output).
        latched_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pose_pub = self.create_publisher(
            PoseStamped, "~/grasp_pose", latched_qos
        )

        self._srv = self.create_service(
            Trigger,
            "~/execute_sock_grasp",
            self._on_trigger,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            "mobile_grasp_coordinator ready. Trigger ~/execute_sock_grasp."
        )

    # ------------------------------------------------------------------
    # Service callback (the state machine)
    # ------------------------------------------------------------------

    def _on_trigger(self, _request, response):
        target_frame = self.get_parameter("target_frame").value
        world_frame = self.get_parameter("world_frame").value
        arm_reach = float(self.get_parameter("arm_reach").value)
        arm_base = list(self.get_parameter("arm_base_xy").value)
        ax = float(arm_base[0]) if len(arm_base) > 0 else 0.06
        ay = float(arm_base[1]) if len(arm_base) > 1 else 0.0
        approach_standoff = float(self.get_parameter("approach_standoff").value)

        self._grasp_pose_world = None

        # ---- 1. SEGMENT -------------------------------------------------
        self._log_state("SEGMENT")
        sock = self._segment(target_frame)
        if sock is None:
            return self._fail(response, "SEGMENT: no sock found / segmentation unavailable.")

        cx, cy, cz = (
            sock.centroid.point.x,
            sock.centroid.point.y,
            sock.centroid.point.z,
        )
        self.get_logger().info(
            f"SEGMENT: sock '{sock.label}' score={sock.score:.2f} "
            f"centroid=({cx:.3f},{cy:.3f},{cz:.3f}) in {target_frame}."
        )

        # Build the top-down grasp pose in base_link (as the GRASP step always
        # did), then remember it in the world frame while the sock is visible —
        # this is the open-loop target we'll grasp after driving up.
        grasp_pose_base = self._make_grasp_pose(target_frame, cx, cy, cz)
        grasp_pose_world = self._to_frame(grasp_pose_base, world_frame)
        if grasp_pose_world is None:
            return self._fail(
                response,
                f"SEGMENT: TF {target_frame}->{world_frame} unavailable; "
                "cannot remember grasp pose.",
            )
        self._grasp_pose_world = grasp_pose_world
        self.get_logger().info(
            f"SEGMENT: stored grasp pose in '{world_frame}' at "
            f"({grasp_pose_world.pose.position.x:.3f},"
            f"{grasp_pose_world.pose.position.y:.3f},"
            f"{grasp_pose_world.pose.position.z:.3f})."
        )

        # ---- 2. REACH_CHECK --------------------------------------------
        self._log_state("REACH_CHECK")
        horiz = math.hypot(cx - ax, cy - ay)
        reachable = horiz <= arm_reach
        self.get_logger().info(
            f"REACH_CHECK: horizontal distance from arm mount ({ax:.3f},{ay:.3f}) "
            f"= {horiz:.3f}m, arm_reach={arm_reach:.3f}m -> "
            f"{'reachable (skip APPROACH)' if reachable else 'out of reach'}."
        )

        # ---- 3. APPROACH (conditional) ---------------------------------
        if not reachable:
            self._log_state("APPROACH")
            ok, msg = self._approach((cx, cy, cz), sock.centroid.header.frame_id,
                                     approach_standoff)
            if not ok:
                return self._fail(response, f"APPROACH: {msg}")
            self.get_logger().info(f"APPROACH: {msg}")

        # ---- 4. GRASP --------------------------------------------------
        self._log_state("GRASP")
        grasp_mode = self.get_parameter("grasp_mode").value
        if grasp_mode == "preset":
            # Validated path for this 4-DOF arm: a free-form Cartesian grasp pose
            # is infeasible at floor level (wrist self-collides with the arm-mounted
            # camera -> OMPL cannot sample a valid IK state). The APPROACH step has
            # centred the sock at the standoff, so we fire grasp_server's tuned
            # PRESET reach (grasp_pre -> grasp_reach -> close), which plans as joint
            # goals and is RViz-validated. No pose is sent.
            self.get_logger().info("GRASP: preset reach (base centred the sock at standoff).")
            ok, msg = self._grasp(None)
        else:
            # 'pose' mode (e.g. elevated/reachable targets): recover the remembered
            # world-frame grasp pose into base_link and send it as target_pose.
            grasp_pose = self._from_world(self._grasp_pose_world, target_frame)
            if grasp_pose is None:
                return self._fail(
                    response,
                    f"GRASP: TF {world_frame}->{target_frame} unavailable; "
                    "cannot recover remembered grasp pose.",
                )
            gp = grasp_pose.pose.position
            self.get_logger().info(
                f"GRASP: recovered grasp pose in '{target_frame}' at "
                f"({gp.x:.3f},{gp.y:.3f},{gp.z:.3f})."
            )
            self._pose_pub.publish(grasp_pose)
            ok, msg = self._grasp(grasp_pose)
        if not ok:
            return self._fail(response, f"GRASP: {msg}")

        # ---- 5. DONE ----------------------------------------------------
        self._log_state("DONE")
        response.success = True
        # Report the sock centroid (cx,cy,cz) — always in scope. The earlier
        # gx/gy/gz here were never assigned (NameError on the first successful
        # grasp); preset mode sends no pose, so the centroid is the meaningful
        # grasp location to report.
        response.message = (
            f"Sock grasp complete. {'approached then ' if not reachable else ''}"
            f"grasped at ({cx:.3f},{cy:.3f},{cz:.3f}) in {target_frame}. {msg}"
        )
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    # Per-step orchestration helpers (each returns gracefully on failure)
    # ------------------------------------------------------------------

    def _segment(self, target_frame):
        """Call /segment_socks once; return the SockCloud or None on any failure."""
        timeout_s = float(self.get_parameter("segment_timeout_s").value)
        if not self._seg_client.wait_for_server(timeout_sec=timeout_s):
            self.get_logger().warn(
                f"Segmentation server '{self.get_parameter('segment_action').value}' "
                "not available."
            )
            return None

        goal = SegmentSocks.Goal()
        goal.target_frame = target_frame
        goal.min_score = float(self.get_parameter("min_score").value)
        goal.max_range = float(self.get_parameter("max_range").value)
        goal.publish_debug = False

        result = self._send_and_wait(self._seg_client, goal, timeout_s, "segment")
        if result is None:
            return None
        if not result.found:
            self.get_logger().info("Segmentation returned found=false.")
            return None
        return result.sock

    def _approach(self, centroid_xyz, frame_id, standoff):
        """Call /approach_target; return (ok, message)."""
        timeout_s = float(self.get_parameter("approach_timeout_s").value)
        if not self._approach_client.wait_for_server(timeout_sec=timeout_s):
            return False, (
                f"approach server '{self.get_parameter('approach_action').value}' "
                "not available."
            )

        goal = ApproachTarget.Goal()
        pt = PointStamped()
        pt.header.frame_id = frame_id or self.get_parameter("target_frame").value
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y, pt.point.z = (
            float(centroid_xyz[0]), float(centroid_xyz[1]), float(centroid_xyz[2]),
        )
        goal.target = pt
        goal.standoff = float(standoff)
        goal.timeout = timeout_s

        result = self._send_and_wait(self._approach_client, goal, timeout_s, "approach")
        if result is None:
            return False, "approach goal rejected or timed out."
        return bool(result.success), (
            result.message or f"final_distance={result.final_distance:.3f}m"
        )

    def _grasp(self, grasp_pose):
        """Call /grasp_object; return (ok, message).

        grasp_pose=None -> PRESET grasp (empty target_pose: grasp_server runs its
        tuned grasp_pre/grasp_reach joint sequence). A PoseStamped -> pose-targeted.
        """
        timeout_s = float(self.get_parameter("grasp_timeout_s").value)
        if not self._grasp_client.wait_for_server(timeout_sec=timeout_s):
            return False, (
                f"grasp server '{self.get_parameter('grasp_action').value}' "
                "not available."
            )

        goal = GraspObject.Goal()
        if grasp_pose is not None:
            goal.target_pose = grasp_pose  # pose mode; empty frame_id => preset
        goal.approach_height = 0.0  # let grasp_server use its default standoff

        result = self._send_and_wait(self._grasp_client, goal, timeout_s, "grasp")
        if result is None:
            return False, "grasp goal rejected or timed out."
        return bool(result.success), result.message or ""

    def _make_grasp_pose(self, frame_id, cx, cy, cz) -> PoseStamped:
        """Top-down PoseStamped at the sock centroid (yaw 0 -> pure roll-pi)."""
        quat = top_down_quaternion(0.0)
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(cx)
        pose.pose.position.y = float(cy)
        pose.pose.position.z = float(cz)
        pose.pose.orientation = Quaternion(
            x=quat[0], y=quat[1], z=quat[2], w=quat[3]
        )
        return pose

    # ------------------------------------------------------------------
    # TF helpers (store grasp pose in world_frame; retrieve into base_link)
    # ------------------------------------------------------------------

    def _to_frame(self, pose: PoseStamped, target_frame: str):
        """Transform *pose* into *target_frame* at its own stamp; None on failure."""
        try:
            return self._tf_buffer.transform(
                pose, target_frame, timeout=Duration(seconds=0.5)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
            tf2_ros.TransformException,
        ) as exc:
            self.get_logger().warn(
                f"TF transform {pose.header.frame_id}->{target_frame} failed: {exc}"
            )
            return None

    def _from_world(self, pose_world: PoseStamped, target_frame: str):
        """Transform the stored world-frame pose into *target_frame* at latest TF.

        The pose is world-fixed in ``world_frame``, so we stamp it with the
        latest available time (``rclpy.time.Time(0)``) and let tf2 use the most
        recent transform — the base may have moved since the pose was captured.
        """
        pose = PoseStamped()
        pose.header.frame_id = pose_world.header.frame_id
        pose.header.stamp = rclpy.time.Time().to_msg()
        pose.pose = pose_world.pose
        return self._to_frame(pose, target_frame)

    # ------------------------------------------------------------------
    # Generic action send/wait (mirrors grasp_pose_node._call_segment)
    # ------------------------------------------------------------------

    def _send_and_wait(self, client, goal, timeout_s, label):
        """Send *goal* on *client*, spin until result/timeout. Returns Result or None."""
        send_future = client.send_goal_async(goal)
        if not self._spin_until(send_future, timeout_s):
            self.get_logger().warn(f"{label}: timed out waiting for goal acceptance.")
            return None

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(f"{label}: goal was rejected.")
            return None

        result_future = goal_handle.get_result_async()
        if not self._spin_until(result_future, timeout_s):
            self.get_logger().warn(f"{label}: timed out waiting for result.")
            return None

        wrapped = result_future.result()
        return wrapped.result if wrapped is not None else None

    def _spin_until(self, future, timeout_s: float) -> bool:
        """Block until *future* completes or *timeout_s* elapses (executor spins)."""
        import time

        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and not future.done():
            if self.get_clock().now().nanoseconds > deadline:
                return False
            time.sleep(0.02)
        return future.done()

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    def _log_state(self, state: str) -> None:
        self.get_logger().info(f"[state] -> {state}")

    def _fail(self, response, message: str):
        self._log_state("FAILED")
        response.success = False
        response.message = message
        self.get_logger().warn(message)
        return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = MobileGraspCoordinator()
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
