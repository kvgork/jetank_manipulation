#!/usr/bin/env python3
"""
Pure grasp math for the JeTank arm — no ROS types, unit-testable.

Shared by ``grasp_pose_node`` (Trigger-driven grasp-pose planner / debug tool)
and ``mobile_grasp_coordinator`` (Phase 7 mobile-manipulation FSM) so the
segment-cloud -> top-down grasp pose geometry lives in exactly one place.

Orientation convention: top-down approach — the gripper's approach axis (tool
+Z) points DOWN, i.e. along -Z of the target frame. That is a rotation of
roll=pi about X. The grasp YAW (rotation about the target-frame Z) is composed
on top. The final quaternion is built manually with numpy/math (neither
tf_transformations nor transforms3d are available in this RoboStack Humble
pixi env — verified).

Tested by ``test/test_grasp_pose.py``.
"""

import math
from typing import Optional, Tuple

import numpy as np


def wrap_to_half_pi(angle: float) -> float:
    """
    Wrap an angle into (-pi/2, pi/2].

    A grasp yaw and yaw+pi command an identical (symmetric) gripper opening, so
    we normalise into a half-open quadrant to keep the value sane/comparable.
    """
    a = (angle + math.pi / 2.0) % math.pi - math.pi / 2.0
    # math fmod/% can land exactly on -pi/2; fold it up to +pi/2 for stability.
    if a <= -math.pi / 2.0:
        a += math.pi
    return a


def pca_long_axis_yaw(points_xy: np.ndarray) -> Optional[float]:
    """
    Return the angle (rad) of the principal (longest) axis of an XY cloud.

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
    """
    Compute the grasp position (x, y, z) and yaw (rad) for a top-down grasp.

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
    """
    Quaternion (x, y, z, w) for a top-down grasp at the given yaw.

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
