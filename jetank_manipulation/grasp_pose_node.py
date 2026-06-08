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

The final quaternion is built manually with numpy (neither tf_transformations
nor transforms3d are available in this RoboStack Humble pixi env — verified).

Usage:
  ros2 run jetank_manipulation grasp_pose_node --ros-args -p use_sim_time:=true
  ros2 service call /grasp_pose_node/plan_sock_grasp std_srvs/srv/Trigger '{}'
"""

import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Quaternion
from visualization_msgs.msg import Marker

from sensor_msgs_py import point_cloud2

from jetank_detection.action import SegmentSocks


# ---------------------------------------------------------------------------
# Pure grasp math (no ROS) — unit-testable
# ---------------------------------------------------------------------------


def wrap_to_half_pi(angle: float) -> float:
    """Wrap an angle into (-pi/2, pi/2].

    A grasp yaw and yaw+pi command an identical (symmetric) gripper opening, so
    we normalise into a half-open quadrant to keep the value sane/comparable.
    """
    a = (angle + math.pi / 2.0) % math.pi - math.pi / 2.0
    # math fmod/% can land exactly on -pi/2; fold it up to +pi/2 for stability.
    if a <= -math.pi / 2.0:
        a += math.pi
    return a


def pca_long_axis_yaw(points_xy: np.ndarray) -> Optional[float]:
    """Return the angle (rad) of the principal (longest) axis of an XY cloud.

    ``points_xy`` is an (N, 2) array. Returns ``atan2(v_y, v_x)`` of the
    eigenvector with the largest eigenvalue of the XY covariance, or ``None``
    when the footprint is degenerate (fewer than 2 distinct points / zero
    spread) so the caller can fall back to a default yaw.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return None

    centered = pts - pts.mean(axis=0)
    spread = np.linalg.norm(centered, axis=1)
    if not np.any(spread > 1e-9):
        return None

    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None

    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    principal = eigvecs[:, int(np.argmax(eigvals))]  # longest axis
    return math.atan2(float(principal[1]), float(principal[0]))


def compute_grasp_fields(
    points: np.ndarray,
    centroid: Tuple[float, float, float],
    dimensions: Tuple[float, float, float],
    yaw_from_pca: bool = True,
    approach_z_offset: float = 0.0,
    default_yaw: float = 0.0,
) -> Tuple[Tuple[float, float, float], float]:
    """Compute the grasp position (x, y, z) and yaw (rad) for a top-down grasp.

    Pure function — no ROS types — so it is trivially unit-testable.

    Parameters
    ----------
    points:
        (N, 3) array of cloud points in the target frame. May be empty.
    centroid:
        (cx, cy, cz) blob centroid in the target frame.
    dimensions:
        (dx, dy, dz) AABB size (unused for the math today but kept in the
        signature so callers pass the full SockCloud payload and future
        heuristics — e.g. width-based yaw — slot in without an API change).
    yaw_from_pca:
        If True, yaw = PCA-long-axis(theta) + pi/2 (gripper opening
        perpendicular to the sock). If False (or PCA degenerate), yaw =
        default_yaw.
    approach_z_offset:
        Added to the top-of-blob z to give the final grasp z (a top-down grasp
        descends onto the sock from above).
    default_yaw:
        Fallback yaw when not using / unable to use PCA.

    Returns
    -------
    ((x, y, z), yaw)
        Position in the target frame and grasp yaw about the target-frame Z.
    """
    cx, cy, cz = (float(centroid[0]), float(centroid[1]), float(centroid[2]))

    pts = np.asarray(points, dtype=np.float64)
    have_points = pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 3

    # z = top of the blob (max point z) + approach offset; fall back to centroid
    # z when no points are available.
    if have_points:
        top_z = float(np.max(pts[:, 2]))
    else:
        top_z = cz
    z = top_z + float(approach_z_offset)

    # yaw
    yaw = float(default_yaw)
    if yaw_from_pca and have_points:
        theta = pca_long_axis_yaw(pts[:, :2])
        if theta is not None:
            # gripper opening perpendicular to the sock's long axis
            yaw = wrap_to_half_pi(theta + math.pi / 2.0)

    return (cx, cy, z), yaw


def top_down_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a top-down grasp at the given yaw.

    The orientation is the composition (target-frame Z yaw) * (roll = pi about
    X). The roll-pi flips the tool's +Z to point DOWN (approach along the
    target frame -Z); the yaw then spins the gripper about the (now vertical)
    approach axis. Computed manually with numpy — no transforms3d /
    tf_transformations in this env.

    Quaternion convention: q = q_yaw * q_roll, Hamilton product, (x, y, z, w).
    """
    # q_yaw: rotation by yaw about Z
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    qz = (0.0, 0.0, sy, cy)
    # q_roll: rotation by pi about X -> (sin(pi/2), 0, 0, cos(pi/2)) = (1, 0, 0, 0)
    cr, sr = math.cos(math.pi / 2.0), math.sin(math.pi / 2.0)
    qx = (sr, 0.0, 0.0, cr)
    return _quat_mul(qz, qx)


def _quat_mul(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Hamilton product q1 * q2, with quaternions as (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
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
        latched_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pose_pub = self.create_publisher(
            PoseStamped, "~/grasp_pose", latched_qos
        )
        self._marker_pub = self.create_publisher(
            Marker, "~/grasp_pose_marker", latched_qos
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

        goal = SegmentSocks.Goal()
        goal.target_frame = target_frame
        goal.min_score = min_score
        goal.max_range = max_range
        goal.publish_debug = False

        self.get_logger().info(
            f"Requesting sock segmentation (frame={target_frame}, "
            f"min_score={min_score}, max_range={max_range})..."
        )

        result = self._call_segment(goal, timeout_s)
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
        points = self._cloud_to_xyz(sock.cloud)

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

    def _call_segment(self, goal, timeout_s: float):
        """Send the goal and block (executor spins) until result or timeout.

        Returns the action Result, or None on rejection/timeout.
        """
        send_future = self._seg_client.send_goal_async(goal)
        if not self._spin_until(send_future, timeout_s):
            self.get_logger().warn("Timed out waiting for goal acceptance.")
            return None

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Segmentation goal was rejected.")
            return None

        result_future = goal_handle.get_result_async()
        if not self._spin_until(result_future, timeout_s):
            self.get_logger().warn("Timed out waiting for segmentation result.")
            return None

        wrapped = result_future.result()
        return wrapped.result if wrapped is not None else None

    def _spin_until(self, future, timeout_s: float) -> bool:
        """Block until *future* completes or *timeout_s* elapses.

        Uses a deadline on the node clock; the surrounding MultiThreadedExecutor
        continues spinning callbacks (the service runs in a reentrant group).
        """
        deadline = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while rclpy.ok() and not future.done():
            if self.get_clock().now().nanoseconds > deadline:
                return False
            # Yield briefly; another executor thread services the action.
            self._sleep(0.02)
        return future.done()

    @staticmethod
    def _sleep(seconds: float) -> None:
        import time

        time.sleep(seconds)

    @staticmethod
    def _cloud_to_xyz(cloud) -> np.ndarray:
        """Read a PointCloud2 into an (N, 3) float64 numpy array (x, y, z)."""
        try:
            structured = point_cloud2.read_points(
                cloud, field_names=("x", "y", "z"), skip_nans=True
            )
        except Exception:
            return np.empty((0, 3), dtype=np.float64)

        arr = np.asarray(structured)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        # read_points returns a structured array; stack the fields into (N, 3).
        xyz = np.column_stack(
            (
                np.asarray(arr["x"], dtype=np.float64),
                np.asarray(arr["y"], dtype=np.float64),
                np.asarray(arr["z"], dtype=np.float64),
            )
        )
        return xyz

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
    rclpy.init(args=args)
    node = GraspPoseNode()
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
