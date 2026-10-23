"""Shared XYZW-quaternion math for the mimic Warp kernels.

Isaac Lab stores quaternions in (x, y, z, w) order while Warp's built-in
``wp.quatf`` types assume (w, x, y, z).  Instead of shuffling data at every
kernel boundary, the mimic Warp terms snapshot everything into plain float32
buffers and do the quaternion algebra by hand in XYZW order -- mirroring the
torch terms exactly.
"""

from __future__ import annotations

import warp as wp


@wp.func
def quat_mul_xyzw(q1: wp.vec4f, q2: wp.vec4f) -> wp.vec4f:
    """Hamilton product ``q1 * q2`` for (x, y, z, w) quaternions."""
    x = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1]
    y = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0]
    z = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3]
    w = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
    return wp.vec4f(x, y, z, w)


@wp.func
def quat_conj_xyzw(q: wp.vec4f) -> wp.vec4f:
    """Conjugate of a (x, y, z, w) quaternion."""
    return wp.vec4f(-q[0], -q[1], -q[2], q[3])


@wp.func
def quat_rotate_xyzw(q: wp.vec4f, v: wp.vec3f) -> wp.vec3f:
    """Rotate vector ``v`` by unit quaternion ``q``: ``v + 2w(u x v) + 2u x (u x v)``."""
    u = wp.vec3f(q[0], q[1], q[2])
    t = 2.0 * wp.cross(u, v)
    return v + q[3] * t + wp.cross(u, t)


@wp.func
def quat_rotate_inv_xyzw(q: wp.vec4f, v: wp.vec3f) -> wp.vec3f:
    """Rotate vector ``v`` by the inverse of unit quaternion ``q``."""
    return quat_rotate_xyzw(quat_conj_xyzw(q), v)


@wp.func
def yaw_quat_xyzw(q: wp.vec4f) -> wp.vec4f:
    """Yaw-only quaternion, mirroring ``isaaclab.utils.math.yaw_quat`` (XYZW).

    Same atan2 formula and the same final normalize (eps-clamped) as the torch
    implementation, so the relative-transform math matches numerically.
    """
    yaw = wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
    z = wp.sin(yaw * 0.5)
    w = wp.cos(yaw * 0.5)
    n = wp.max(wp.sqrt(z * z + w * w), 1.0e-12)
    return wp.vec4f(0.0, 0.0, z / n, w / n)


@wp.func
def quat_error_magnitude_xyzw(q1: wp.vec4f, q2: wp.vec4f) -> float:
    """Rotation angle between ``q1`` and ``q2`` in radians.

    Mirrors ``isaaclab.utils.math.quat_error_magnitude`` (quat_box_minus +
    axis_angle_from_quat) branch for branch, including the small-angle Taylor
    approximation, so the Warp rewards match the torch rewards numerically.
    """
    qd = quat_mul_xyzw(q1, quat_conj_xyzw(q2))
    # canonicalize the sign (axis_angle_from_quat negates the quat when w < 0)
    if qd[3] < 0.0:
        qd = wp.vec4f(-qd[0], -qd[1], -qd[2], -qd[3])
    mag = wp.sqrt(qd[0] * qd[0] + qd[1] * qd[1] + qd[2] * qd[2])
    half_angle = wp.atan2(mag, qd[3])
    angle = 2.0 * half_angle
    if wp.abs(angle) > 1.0e-6:
        sin_half_over_angle = wp.sin(half_angle) / angle
    else:
        sin_half_over_angle = 0.5 - angle * angle / 48.0
    ex = qd[0] / sin_half_over_angle
    ey = qd[1] / sin_half_over_angle
    ez = qd[2] / sin_half_over_angle
    return wp.sqrt(ex * ex + ey * ey + ez * ez)
