#!/usr/bin/env python3
"""
Shared ROS plumbing for jetank_manipulation nodes.

Consolidates the action-client send/spin-until-result helpers, the SegmentSocks
goal construction, the latched QoS profile, and the PointCloud2 -> numpy
conversion that were previously copy-pasted between ``grasp_pose_node`` and
``mobile_grasp_coordinator``. Pure grasp geometry lives in ``grasp_math``.

All helpers assume the calling node spins under a MultiThreadedExecutor with
its callbacks in a ReentrantCallbackGroup (the package-wide idiom): the
blocking waiters below poll a future while another executor thread services
the underlying action client.
"""

import time

import numpy as np

import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from sensor_msgs_py import point_cloud2

from jetank_detection.action import SegmentSocks


def latched_qos(depth: int = 1) -> QoSProfile:
    """
    Latched (transient-local, keep-last) QoS.

    Late subscribers (e.g. RViz) immediately receive the most recent message.
    """
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def make_segment_goal(
    target_frame: str,
    min_score: float,
    max_range: float,
    publish_debug: bool = False,
) -> "SegmentSocks.Goal":
    """Build a jetank_detection SegmentSocks goal from the common parameters."""
    goal = SegmentSocks.Goal()
    goal.target_frame = target_frame
    goal.min_score = float(min_score)
    goal.max_range = float(max_range)
    goal.publish_debug = bool(publish_debug)
    return goal


def spin_until_complete(node, future, timeout_s: float) -> bool:
    """
    Block until *future* completes or *timeout_s* elapses.

    Uses a deadline on the node clock; the surrounding MultiThreadedExecutor
    continues spinning callbacks (the caller runs in a reentrant group), so
    another executor thread services the action while we poll.
    """
    deadline = node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
    while rclpy.ok() and not future.done():
        if node.get_clock().now().nanoseconds > deadline:
            return False
        # Yield briefly; another executor thread services the action.
        time.sleep(0.02)
    return future.done()


def send_goal_and_wait(node, client, goal, timeout_s: float, label: str):
    """
    Send *goal* on *client*, spin until result/timeout.

    Returns the action Result, or None on rejection/timeout (a warning naming
    *label* is logged on the node).
    """
    send_future = client.send_goal_async(goal)
    if not spin_until_complete(node, send_future, timeout_s):
        node.get_logger().warn(f"{label}: timed out waiting for goal acceptance.")
        return None

    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        node.get_logger().warn(f"{label}: goal was rejected.")
        return None

    result_future = goal_handle.get_result_async()
    if not spin_until_complete(node, result_future, timeout_s):
        node.get_logger().warn(f"{label}: timed out waiting for result.")
        return None

    wrapped = result_future.result()
    return wrapped.result if wrapped is not None else None


def cloud_to_xyz(cloud) -> np.ndarray:
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
