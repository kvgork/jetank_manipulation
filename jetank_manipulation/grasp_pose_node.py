#!/usr/bin/env python3
"""Grasp-pose planning node for the JeTank arm.

Computes a *top-down* grasp pose from a segmented sock point-cloud blob. It does
**not** move the arm — it only outputs a ``geometry_msgs/PoseStamped`` and an
RViz ``visualization_msgs/Marker`` (ARROW) showing the grasp approach.

Pipeline (on Trigger):
  ~/plan_sock_grasp (std_srvs/srv/Trigger)
    -> ActionClient /segment_socks (jetank_detection/action/SegmentSocks)
    -> read result.sock (jetank_detection/SockCloud), cloud already in target_frame
    -> compute_grasp_fields(points, centroid, dims, ...) -> position xyz + yaw
    -> publish PoseStamped on ~/grasp_pose (transient-local, keep-last)
    -> publish Marker (ARROW) on ~/grasp_pose_marker

Orientation: top-down approach — the gripper's approach axis (tool +Z) points
DOWN, i.e. along -Z of the target frame. That is a rotation of roll=pi about X.
The grasp YAW (rotation about the target-frame Z) is then composed on top:
  - yaw_from_pca=true  : PCA of the cloud's XY footprint gives the sock's long
                         axis angle theta; the gripper opening is placed
                         PERPENDICULAR to that axis, so grasp yaw = theta + pi/2.
  - yaw_from_pca=false : fixed yaw = default_yaw param.

The pure grasp math (PCA yaw, top-down quaternion) lives in the shared
``jetank_manipulation.grasp_math`` module; the action-client plumbing lives in
``jetank_manipulation.action_utils``. Both are shared with
``mobile_grasp_coordinator``.

Usage:
  ros2 run jetank_manipulation grasp_pose_node --ros-args -p use_sim_time:=true
  ros2 service call /grasp_pose_node/plan_sock_grasp std_srvs/srv/Trigger '{}'
"""

import math

from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Quaternion
from visualization_msgs.msg import Marker

from jetank_detection.action import SegmentSocks

from jetank_manipulation.action_utils import (
    cloud_to_xyz,
    latched_qos,
    make_segment_goal,
    send_goal_and_wait,
)
# The math names are re-exported here (noqa) for backward compatibility:
# test/test_grasp_pose.py and historical callers access them via this module.
from jetank_manipulation.grasp_math import (  # noqa: F401
    compute_grasp_fields,
    pca_long_axis_yaw,
    top_down_quaternion,
    wrap_to_half_pi,
)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class GraspPoseNode(Node):
    """Trigger -> /segment_socks -> compute + publish a top-down grasp pose."""

    def __init__(self) -> None:
        super().__init__("grasp_pose_node")

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("min_score", 0.3)
        self.declare_parameter("max_range", 3.0)
        self.declare_parameter("yaw_from_pca", True)
        self.declare_parameter("approach_z_offset", 0.0)
        self.declare_parameter("default_yaw", 0.0)
        self.declare_parameter("segment_action", "/segment_socks")
        self.declare_parameter("result_timeout_s", 10.0)

        self._cb_group = ReentrantCallbackGroup()

        action_name = self.get_parameter("segment_action").value
        self._seg_client = ActionClient(
            self,
            SegmentSocks,
            action_name,
            callback_group=self._cb_group,
        )

        # Latched (transient-local, keep-last) so RViz / late subscribers see
        # the most recent grasp pose immediately.
        latched = latched_qos()
        self._pose_pub = self.create_publisher(
            PoseStamped, "~/grasp_pose", latched
        )
        self._marker_pub = self.create_publisher(
            Marker, "~/grasp_pose_marker", latched
        )

        self._srv = self.create_service(
            Trigger,
            "~/plan_sock_grasp",
            self._on_trigger,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"grasp_pose_node ready. Trigger ~/plan_sock_grasp -> {action_name}"
        )

    # ------------------------------------------------------------------
    # Service callback
    # ------------------------------------------------------------------

    def _on_trigger(self, _request, response):
        target_frame = self.get_parameter("target_frame").value
        min_score = float(self.get_parameter("min_score").value)
        max_range = float(self.get_parameter("max_range").value)
        yaw_from_pca = bool(self.get_parameter("yaw_from_pca").value)
        approach_z_offset = float(self.get_parameter("approach_z_offset").value)
        default_yaw = float(self.get_parameter("default_yaw").value)
        timeout_s = float(self.get_parameter("result_timeout_s").value)

        if not self._seg_client.wait_for_server(timeout_sec=timeout_s):
            response.success = False
            response.message = (
                f"Segmentation action server '{self.get_parameter('segment_action').value}' "
                "not available."
            )
            self.get_logger().warn(response.message)
            return response

        goal = make_segment_goal(target_frame, min_score, max_range)

        self.get_logger().info(
            f"Requesting sock segmentation (frame={target_frame}, "
            f"min_score={min_score}, max_range={max_range})..."
        )

        result = send_goal_and_wait(self, self._seg_client, goal, timeout_s, "segment")
        if result is None:
            response.success = False
            response.message = (
                "Segmentation goal failed: rejected or timed out "
                f"after {timeout_s:.1f}s."
            )
            self.get_logger().warn(response.message)
            return response

        if not result.found:
            response.success = False
            response.message = "No sock found by segmentation (result.found=false)."
            self.get_logger().info(response.message)
            return response

        sock = result.sock
        centroid = (
            sock.centroid.point.x,
            sock.centroid.point.y,
            sock.centroid.point.z,
        )
        dims = (sock.dimensions.x, sock.dimensions.y, sock.dimensions.z)

        # Read the cloud into an (N, 3) numpy array (target_frame coords).
        points = cloud_to_xyz(sock.cloud)

        (gx, gy, gz), yaw = compute_grasp_fields(
            points,
            centroid,
            dims,
            yaw_from_pca=yaw_from_pca,
            approach_z_offset=approach_z_offset,
            default_yaw=default_yaw,
        )
        quat = top_down_quaternion(yaw)

        pose = PoseStamped()
        pose.header.frame_id = target_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = gx
        pose.pose.position.y = gy
        pose.pose.position.z = gz
        pose.pose.orientation = Quaternion(
            x=quat[0], y=quat[1], z=quat[2], w=quat[3]
        )

        self._pose_pub.publish(pose)
        self._marker_pub.publish(self._make_marker(pose))

        yaw_mode = "pca" if yaw_from_pca else "fixed"
        response.success = True
        response.message = (
            f"Grasp pose ({target_frame}): x={gx:.3f} y={gy:.3f} z={gz:.3f} "
            f"yaw={yaw:.3f}rad ({math.degrees(yaw):.1f}deg) mode={yaw_mode} "
            f"label='{sock.label}' score={sock.score:.2f} "
            f"npts={0 if points.size == 0 else points.shape[0]}"
        )
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_marker(pose: PoseStamped) -> Marker:
        """ARROW marker pointing along the grasp approach for RViz."""
        m = Marker()
        m.header = pose.header
        m.ns = "grasp_pose"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose = pose.pose
        # Arrow geometry: length (x), shaft dia (y), head dia (z).
        m.scale.x = 0.10
        m.scale.y = 0.01
        m.scale.z = 0.02
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0
        return m


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None):
    # Imported at call time so bare-env tests can load this module by file
    # path with only the imports above stubbed.
    from jetank_manipulation.node_runner import spin_node

    spin_node(GraspPoseNode, args)


if __name__ == "__main__":
    main()
