"""Warp-consumable variant of :class:`~basic_loco_lib.tasks.mimic.mdp.commands.MotionCommand`.

The classic term does all of its per-step work in torch: ``_update_command``
advances ``time_steps``, gathers the reference frame through lazy properties,
builds the relative body targets with a chain of repeats and quaternion ops, and
``_update_metrics`` evaluates ~12 error metrics -- several dozen kernel launches
plus fresh allocations every step, plus two metrics tensors created lazily each
call.  Only the reset/overflow resample (multinomial adaptive sampling, writes
to sim) genuinely needs torch.

This subclass keeps the classic resample path but replaces the steady-state
update with two Warp kernels operating on persistent buffers:

* ``_metrics_advance_kernel`` -- evaluates the metrics on the pre-advance frame
  (exactly where ``CommandTerm.compute`` calls ``_update_metrics``) and then
  advances ``time_steps`` in place, so the subsequent overflow check and
  ``_adaptive_sampling`` see the same values as the torch version.
* ``_reference_update_kernel`` -- gathers the reference frame directly from the
  raw motion arrays, snapshots the robot state, computes the relative targets
  (``yaw_quat(robot_anchor * anchor^-1)`` chain) and writes them into the
  *canonical* ``body_pos/quat_relative_w`` tensors, so every torch reader (base
  metrics consumers, parity checks) and every Warp kernel sees the same storage.

Motion-end overflow (``time_steps >= total``) is detected from a host-side
numpy mirror of ``time_steps`` -- the tensor only changes inside the advance
kernel and the torch resample path, both of which keep the mirror in sync -- so
the steady state performs no device readback at all.  On overflow (and on
resets) the code falls back to the classic torch ``_resample_command`` and then
re-runs the reference kernel so the buffers are fresh for any observation
computed right after a reset.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from basic_loco_lib.tasks.mimic.mdp.commands import MotionCommand, MotionCommandCfg
from basic_loco_lib.tasks.mimic.mdp_warp.obs_history import register_env_hint
from basic_loco_lib.tasks.mimic.mdp_warp.warp_utils import (
    quat_conj_xyzw,
    quat_error_magnitude_xyzw,
    quat_mul_xyzw,
    quat_rotate_xyzw,
    yaw_quat_xyzw,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _metrics_advance_kernel(
    ts: wp.array(dtype=wp.int64),
    mj_pos: wp.array(dtype=wp.float32, ndim=2),
    mj_vel: wp.array(dtype=wp.float32, ndim=2),
    m_pos: wp.array(dtype=wp.float32, ndim=3),
    m_quat: wp.array(dtype=wp.float32, ndim=3),
    m_lin: wp.array(dtype=wp.float32, ndim=3),
    m_ang: wp.array(dtype=wp.float32, ndim=3),
    body_ids: wp.array(dtype=wp.int32),
    anchor_sel: int,
    env_origins: wp.array(dtype=wp.float32, ndim=2),
    r_pos: wp.array(dtype=wp.float32, ndim=3),
    r_quat: wp.array(dtype=wp.float32, ndim=3),
    r_lin: wp.array(dtype=wp.float32, ndim=3),
    r_ang: wp.array(dtype=wp.float32, ndim=3),
    r_jpos: wp.array(dtype=wp.float32, ndim=2),
    r_jvel: wp.array(dtype=wp.float32, ndim=2),
    rel_pos_prev: wp.array(dtype=wp.float32, ndim=3),
    rel_quat_prev: wp.array(dtype=wp.float32, ndim=3),
    out_anchor_pos: wp.array(dtype=wp.float32),
    out_anchor_rot: wp.array(dtype=wp.float32),
    out_anchor_lin: wp.array(dtype=wp.float32),
    out_anchor_ang: wp.array(dtype=wp.float32),
    out_body_pos: wp.array(dtype=wp.float32),
    out_body_rot: wp.array(dtype=wp.float32),
    out_body_lin: wp.array(dtype=wp.float32),
    out_body_ang: wp.array(dtype=wp.float32),
    out_joint_pos: wp.array(dtype=wp.float32),
    out_joint_vel: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    t = int(ts[i])
    anchor_col = body_ids[anchor_sel]

    # -- anchor errors (reference gets env_origins, robot state is already global)
    dx = m_pos[t, anchor_col, 0] + env_origins[i, 0] - r_pos[i, anchor_col, 0]
    dy = m_pos[t, anchor_col, 1] + env_origins[i, 1] - r_pos[i, anchor_col, 1]
    dz = m_pos[t, anchor_col, 2] + env_origins[i, 2] - r_pos[i, anchor_col, 2]
    out_anchor_pos[i] = wp.sqrt(dx * dx + dy * dy + dz * dz)

    q_m = wp.vec4f(m_quat[t, anchor_col, 0], m_quat[t, anchor_col, 1], m_quat[t, anchor_col, 2], m_quat[t, anchor_col, 3])
    q_r = wp.vec4f(r_quat[i, anchor_col, 0], r_quat[i, anchor_col, 1], r_quat[i, anchor_col, 2], r_quat[i, anchor_col, 3])
    out_anchor_rot[i] = quat_error_magnitude_xyzw(q_m, q_r)

    lx = m_lin[t, anchor_col, 0] - r_lin[i, anchor_col, 0]
    ly = m_lin[t, anchor_col, 1] - r_lin[i, anchor_col, 1]
    lz = m_lin[t, anchor_col, 2] - r_lin[i, anchor_col, 2]
    out_anchor_lin[i] = wp.sqrt(lx * lx + ly * ly + lz * lz)

    ax = m_ang[t, anchor_col, 0] - r_ang[i, anchor_col, 0]
    ay = m_ang[t, anchor_col, 1] - r_ang[i, anchor_col, 1]
    az = m_ang[t, anchor_col, 2] - r_ang[i, anchor_col, 2]
    out_anchor_ang[i] = wp.sqrt(ax * ax + ay * ay + az * az)

    # -- body errors: per-body norm, then mean over tracked bodies (torch does
    #    norm(dim=-1).mean(dim=-1)); position/orientation compare against the
    #    relative targets from the previous update, velocities against raw motion
    num_bodies = body_ids.shape[0]
    e_pos = float(0.0)
    e_rot = float(0.0)
    e_lin = float(0.0)
    e_ang = float(0.0)
    for b in range(num_bodies):
        col = body_ids[b]
        px = rel_pos_prev[i, b, 0] - r_pos[i, col, 0]
        py = rel_pos_prev[i, b, 1] - r_pos[i, col, 1]
        pz = rel_pos_prev[i, b, 2] - r_pos[i, col, 2]
        e_pos += wp.sqrt(px * px + py * py + pz * pz)

        q1 = wp.vec4f(rel_quat_prev[i, b, 0], rel_quat_prev[i, b, 1], rel_quat_prev[i, b, 2], rel_quat_prev[i, b, 3])
        q2 = wp.vec4f(r_quat[i, col, 0], r_quat[i, col, 1], r_quat[i, col, 2], r_quat[i, col, 3])
        rot = quat_error_magnitude_xyzw(q1, q2)
        e_rot += rot

        lx = m_lin[t, col, 0] - r_lin[i, col, 0]
        ly = m_lin[t, col, 1] - r_lin[i, col, 1]
        lz = m_lin[t, col, 2] - r_lin[i, col, 2]
        e_lin += wp.sqrt(lx * lx + ly * ly + lz * lz)

        ax = m_ang[t, col, 0] - r_ang[i, col, 0]
        ay = m_ang[t, col, 1] - r_ang[i, col, 1]
        az = m_ang[t, col, 2] - r_ang[i, col, 2]
        e_ang += wp.sqrt(ax * ax + ay * ay + az * az)

    inv_bodies = 1.0 / float(num_bodies)
    out_body_pos[i] = e_pos * inv_bodies
    out_body_rot[i] = e_rot * inv_bodies
    out_body_lin[i] = e_lin * inv_bodies
    out_body_ang[i] = e_ang * inv_bodies

    # -- joint errors
    num_j = mj_pos.shape[1]
    e_jpos = float(0.0)
    e_jvel = float(0.0)
    for j in range(num_j):
        d = mj_pos[t, j] - r_jpos[i, j]
        e_jpos += d * d
        d = mj_vel[t, j] - r_jvel[i, j]
        e_jvel += d * d
    out_joint_pos[i] = wp.sqrt(e_jpos)
    out_joint_vel[i] = wp.sqrt(e_jvel)

    # -- advance the frame index last: metrics above used the pre-advance value,
    #    exactly like the torch version (metrics run before _update_command)
    ts[i] = wp.int64(t + 1)


@wp.kernel
def _reference_update_kernel(
    ts: wp.array(dtype=wp.int64),
    mj_pos: wp.array(dtype=wp.float32, ndim=2),
    mj_vel: wp.array(dtype=wp.float32, ndim=2),
    m_pos: wp.array(dtype=wp.float32, ndim=3),
    m_quat: wp.array(dtype=wp.float32, ndim=3),
    m_lin: wp.array(dtype=wp.float32, ndim=3),
    m_ang: wp.array(dtype=wp.float32, ndim=3),
    body_ids: wp.array(dtype=wp.int32),
    anchor_sel: int,
    env_origins: wp.array(dtype=wp.float32, ndim=2),
    r_pos: wp.array(dtype=wp.float32, ndim=3),
    r_quat: wp.array(dtype=wp.float32, ndim=3),
    r_lin: wp.array(dtype=wp.float32, ndim=3),
    r_ang: wp.array(dtype=wp.float32, ndim=3),
    cmd_buf: wp.array(dtype=wp.float32, ndim=2),
    anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    body_lin: wp.array(dtype=wp.float32, ndim=3),
    body_ang: wp.array(dtype=wp.float32, ndim=3),
    r_anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    r_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    r_body_pos: wp.array(dtype=wp.float32, ndim=3),
    r_body_quat: wp.array(dtype=wp.float32, ndim=3),
    r_body_lin: wp.array(dtype=wp.float32, ndim=3),
    r_body_ang: wp.array(dtype=wp.float32, ndim=3),
    rel_pos: wp.array(dtype=wp.float32, ndim=3),
    rel_quat: wp.array(dtype=wp.float32, ndim=3),
):
    i = wp.tid()
    t = int(ts[i])
    anchor_col = body_ids[anchor_sel]
    num_j = mj_pos.shape[1]
    num_bodies = body_ids.shape[0]

    # -- command buffer: [reference joint_pos | reference joint_vel]
    for j in range(num_j):
        cmd_buf[i, j] = mj_pos[t, j]
        cmd_buf[i, num_j + j] = mj_vel[t, j]

    # -- reference anchor (motion frame + env origins)
    a_px = m_pos[t, anchor_col, 0] + env_origins[i, 0]
    a_py = m_pos[t, anchor_col, 1] + env_origins[i, 1]
    a_pz = m_pos[t, anchor_col, 2] + env_origins[i, 2]
    anchor_pos[i, 0] = a_px
    anchor_pos[i, 1] = a_py
    anchor_pos[i, 2] = a_pz
    anchor_quat[i, 0] = m_quat[t, anchor_col, 0]
    anchor_quat[i, 1] = m_quat[t, anchor_col, 1]
    anchor_quat[i, 2] = m_quat[t, anchor_col, 2]
    anchor_quat[i, 3] = m_quat[t, anchor_col, 3]
    q_a = wp.vec4f(anchor_quat[i, 0], anchor_quat[i, 1], anchor_quat[i, 2], anchor_quat[i, 3])

    # -- robot anchor
    r_anchor_pos[i, 0] = r_pos[i, anchor_col, 0]
    r_anchor_pos[i, 1] = r_pos[i, anchor_col, 1]
    r_anchor_pos[i, 2] = r_pos[i, anchor_col, 2]
    r_anchor_quat[i, 0] = r_quat[i, anchor_col, 0]
    r_anchor_quat[i, 1] = r_quat[i, anchor_col, 1]
    r_anchor_quat[i, 2] = r_quat[i, anchor_col, 2]
    r_anchor_quat[i, 3] = r_quat[i, anchor_col, 3]
    q_r = wp.vec4f(r_anchor_quat[i, 0], r_anchor_quat[i, 1], r_anchor_quat[i, 2], r_anchor_quat[i, 3])

    # -- delta transform (classic: xy from robot anchor, z from reference anchor;
    #    yaw-only orientation error)
    delta_ori = yaw_quat_xyzw(quat_mul_xyzw(q_r, quat_conj_xyzw(q_a)))
    delta_pos = wp.vec3f(r_anchor_pos[i, 0], r_anchor_pos[i, 1], a_pz)

    # -- per-body reference targets and robot snapshots
    for b in range(num_bodies):
        col = body_ids[b]

        body_lin[i, b, 0] = m_lin[t, col, 0]
        body_lin[i, b, 1] = m_lin[t, col, 1]
        body_lin[i, b, 2] = m_lin[t, col, 2]
        body_ang[i, b, 0] = m_ang[t, col, 0]
        body_ang[i, b, 1] = m_ang[t, col, 1]
        body_ang[i, b, 2] = m_ang[t, col, 2]

        r_body_pos[i, b, 0] = r_pos[i, col, 0]
        r_body_pos[i, b, 1] = r_pos[i, col, 1]
        r_body_pos[i, b, 2] = r_pos[i, col, 2]
        r_body_quat[i, b, 0] = r_quat[i, col, 0]
        r_body_quat[i, b, 1] = r_quat[i, col, 1]
        r_body_quat[i, b, 2] = r_quat[i, col, 2]
        r_body_quat[i, b, 3] = r_quat[i, col, 3]
        r_body_lin[i, b, 0] = r_lin[i, col, 0]
        r_body_lin[i, b, 1] = r_lin[i, col, 1]
        r_body_lin[i, b, 2] = r_lin[i, col, 2]
        r_body_ang[i, b, 0] = r_ang[i, col, 0]
        r_body_ang[i, b, 1] = r_ang[i, col, 1]
        r_body_ang[i, b, 2] = r_ang[i, col, 2]

        # relative target: quat_mul(delta_ori, motion_quat) / delta_pos + rot(delta_ori, motion_pos - anchor_pos)
        q_b = wp.vec4f(m_quat[t, col, 0], m_quat[t, col, 1], m_quat[t, col, 2], m_quat[t, col, 3])
        q_rel = quat_mul_xyzw(delta_ori, q_b)
        rel_quat[i, b, 0] = q_rel[0]
        rel_quat[i, b, 1] = q_rel[1]
        rel_quat[i, b, 2] = q_rel[2]
        rel_quat[i, b, 3] = q_rel[3]

        v = wp.vec3f(
            m_pos[t, col, 0] + env_origins[i, 0] - a_px,
            m_pos[t, col, 1] + env_origins[i, 1] - a_py,
            m_pos[t, col, 2] + env_origins[i, 2] - a_pz,
        )
        rv = quat_rotate_xyzw(delta_ori, v)
        rel_pos[i, b, 0] = delta_pos[0] + rv[0]
        rel_pos[i, b, 1] = delta_pos[1] + rv[1]
        rel_pos[i, b, 2] = delta_pos[2] + rv[2]


@wp.kernel
def _bin_ema_kernel(bin_failed: wp.array(dtype=wp.float32), current: wp.array(dtype=wp.float32), alpha: float):
    k = wp.tid()
    bin_failed[k] = alpha * current[k] + (1.0 - alpha) * bin_failed[k]
    current[k] = 0.0


class MotionCommandWarp(MotionCommand):
    """Motion command whose steady-state update runs as Warp kernels."""

    def __init__(self, cfg: MotionCommandWarpCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # the command manager inits BEFORE the observation manager: register
        # the live env so deferred obs-history dims ('command'/'joint'/'action'
        # sentinels) resolve against it at ObservationManager init
        register_env_hint(env)

        num_envs = self.num_envs
        num_bodies = len(cfg.body_names)
        num_joints = self.motion.joint_pos.shape[1]

        def persistent(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.float32, device=self.device)

        # -- persistent snapshots consumed by the obs/reward/termination kernels
        self._command_buf = persistent(num_envs, 2 * num_joints)
        self._anchor_pos_w = persistent(num_envs, 3)
        self._anchor_quat_w = persistent(num_envs, 4)
        self._body_lin_vel_w = persistent(num_envs, num_bodies, 3)
        self._body_ang_vel_w = persistent(num_envs, num_bodies, 3)
        self._robot_anchor_pos_w = persistent(num_envs, 3)
        self._robot_anchor_quat_w = persistent(num_envs, 4)
        self._robot_body_pos_w = persistent(num_envs, num_bodies, 3)
        self._robot_body_quat_w = persistent(num_envs, num_bodies, 4)
        self._robot_body_lin_vel_w = persistent(num_envs, num_bodies, 3)
        self._robot_body_ang_vel_w = persistent(num_envs, num_bodies, 3)

        # -- one-time zero-copy Warp views (stable pointers across steps)
        self.wp_command = wp.from_torch(self._command_buf)
        self.wp_anchor_pos_w = wp.from_torch(self._anchor_pos_w)
        self.wp_anchor_quat_w = wp.from_torch(self._anchor_quat_w)
        self.wp_body_lin_vel_w = wp.from_torch(self._body_lin_vel_w)
        self.wp_body_ang_vel_w = wp.from_torch(self._body_ang_vel_w)
        self.wp_robot_anchor_pos_w = wp.from_torch(self._robot_anchor_pos_w)
        self.wp_robot_anchor_quat_w = wp.from_torch(self._robot_anchor_quat_w)
        self.wp_robot_body_pos_w = wp.from_torch(self._robot_body_pos_w)
        self.wp_robot_body_quat_w = wp.from_torch(self._robot_body_quat_w)
        self.wp_robot_body_lin_vel_w = wp.from_torch(self._robot_body_lin_vel_w)
        self.wp_robot_body_ang_vel_w = wp.from_torch(self._robot_body_ang_vel_w)
        # the base class allocates body_pos/quat_relative_w once and only ever
        # rebinds them inside its torch _update_command -- which this subclass
        # never calls -- so their storage is stable and the kernels can update
        # the canonical tensors in place (torch readers see the same values)
        self.wp_body_pos_rel_w = wp.from_torch(self.body_pos_relative_w)
        self.wp_body_quat_rel_w = wp.from_torch(self.body_quat_relative_w)

        # -- immutable kernel inputs: raw motion arrays (all bodies, robot body
        #    ordering -- MotionLoader slices them with robot indices), the
        #    tracked-body column ids and the per-env origins
        self._wp_motion = (
            wp.from_torch(self.motion.joint_pos),
            wp.from_torch(self.motion.joint_vel),
            wp.from_torch(self.motion._body_pos_w),
            wp.from_torch(self.motion._body_quat_w),
            wp.from_torch(self.motion._body_lin_vel_w),
            wp.from_torch(self.motion._body_ang_vel_w),
        )
        self._wp_body_ids = wp.array(self.body_indexes.tolist(), dtype=wp.int32, device=self.device)
        self._anchor_sel = int(self.motion_anchor_body_index)
        self._wp_env_origins = wp.from_torch(self._env.scene.env_origins)
        self._wp_ts = wp.from_torch(self.time_steps)

        # -- metric outputs: the base class lazily creates error_body_lin/ang_vel
        #    on first _update_metrics; materialize them up front so the kernel
        #    always writes persistent storage
        for key in ("error_body_lin_vel", "error_body_ang_vel"):
            self.metrics.setdefault(key, torch.zeros(num_envs, dtype=torch.float32, device=self.device))
        self._wp_metrics = tuple(
            wp.from_torch(self.metrics[key])
            for key in (
                "error_anchor_pos",
                "error_anchor_rot",
                "error_anchor_lin_vel",
                "error_anchor_ang_vel",
                "error_body_pos",
                "error_body_rot",
                "error_body_lin_vel",
                "error_body_ang_vel",
                "error_joint_pos",
                "error_joint_vel",
            )
        )

        # -- adaptive-sampling EMA state (persistent since this subclass never
        #    rebinds bin_failed_count)
        self._wp_bin_failed = wp.from_torch(self.bin_failed_count)
        self._wp_cur_bin_failed = wp.from_torch(self._current_bin_failed)

        # -- host mirror of time_steps for sync-free overflow detection: the
        #    tensor only changes in the advance kernel (+1, mirrored below) and
        #    in the torch resample path (which syncs the mirror afterwards)
        self._total = int(self.motion.time_step_total)
        self._host_time_steps = self.time_steps.detach().cpu().numpy().copy()

    @property
    def command(self) -> torch.Tensor:
        # persistent buffer (in-place updated by the reference kernel) instead of
        # a fresh torch.cat, so the Warp bridging in
        # isaaclab_experimental.envs.mdp.generated_commands keeps a valid view
        # under CUDA graph capture
        return self._command_buf

    @staticmethod
    def _live_wp(x):
        """Zero-copy Warp view of a live torch tensor / Warp array."""
        if isinstance(x, wp.array):
            return x
        if torch.is_tensor(x):
            if not x.is_contiguous():
                x = x.contiguous()
            return wp.from_torch(x)
        # lazy warp-backed articulation buffers: resolve through torch
        return wp.from_torch(wp.to_torch(x))

    def _robot_wp_views(self):
        d = self.robot.data
        return (
            self._live_wp(d.body_pos_w),
            self._live_wp(d.body_quat_w),
            self._live_wp(d.body_lin_vel_w),
            self._live_wp(d.body_ang_vel_w),
            self._live_wp(d.joint_pos),
            self._live_wp(d.joint_vel),
        )

    def _update_metrics(self):
        # metrics evaluate the pre-advance frame, then the kernel advances
        # time_steps -- the exact ordering of CommandTerm.compute()
        r_pos, r_quat, r_lin, r_ang, r_jpos, r_jvel = self._robot_wp_views()
        wp.launch(
            kernel=_metrics_advance_kernel,
            dim=self.num_envs,
            inputs=[
                self._wp_ts,
                *self._wp_motion,
                self._wp_body_ids,
                self._anchor_sel,
                self._wp_env_origins,
                r_pos,
                r_quat,
                r_lin,
                r_ang,
                r_jpos,
                r_jvel,
                self.wp_body_pos_rel_w,
                self.wp_body_quat_rel_w,
                *self._wp_metrics,
            ],
            device=self.device,
        )

    def _launch_reference_update(self):
        r_pos, r_quat, r_lin, r_ang, _, _ = self._robot_wp_views()
        wp.launch(
            kernel=_reference_update_kernel,
            dim=self.num_envs,
            inputs=[
                self._wp_ts,
                *self._wp_motion,
                self._wp_body_ids,
                self._anchor_sel,
                self._wp_env_origins,
                r_pos,
                r_quat,
                r_lin,
                r_ang,
                self.wp_command,
                self.wp_anchor_pos_w,
                self.wp_anchor_quat_w,
                self.wp_body_lin_vel_w,
                self.wp_body_ang_vel_w,
                self.wp_robot_anchor_pos_w,
                self.wp_robot_anchor_quat_w,
                self.wp_robot_body_pos_w,
                self.wp_robot_body_quat_w,
                self.wp_robot_body_lin_vel_w,
                self.wp_robot_body_ang_vel_w,
                self.wp_body_pos_rel_w,
                self.wp_body_quat_rel_w,
            ],
            device=self.device,
        )

    def _update_command(self):
        # time_steps was advanced on device by the metrics kernel; mirror the
        # advance host-side and detect motion-end overflow without readback
        self._host_time_steps += 1
        overflow = np.nonzero(self._host_time_steps >= self._total)[0]
        if overflow.size > 0:
            # rare path: classic torch resample (adaptive sampling + sim writes),
            # which re-syncs the mirror and refreshes the buffers itself
            env_ids = torch.as_tensor(overflow, dtype=torch.long, device=self.device)
            self._resample_command(env_ids)
        else:
            self._launch_reference_update()

        wp.launch(
            kernel=_bin_ema_kernel,
            dim=self.bin_count,
            inputs=[self._wp_bin_failed, self._wp_cur_bin_failed, self.cfg.adaptive_alpha],
            device=self.device,
        )

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if len(env_ids) > 0:
            self._sync_host_mirror(env_ids)
            # refresh the Warp-visible buffers immediately: the reset path may be
            # followed by observation computation before the next compute() call
            self._launch_reference_update()

    def _sync_host_mirror(self, env_ids):
        ts_host = self.time_steps.detach().cpu().numpy()
        if isinstance(env_ids, slice):
            self._host_time_steps[...] = ts_host
            return
        ids = env_ids.detach().cpu().numpy() if torch.is_tensor(env_ids) else np.asarray(env_ids, dtype=np.int64)
        if ids.size:
            self._host_time_steps[ids] = ts_host[ids]


@configclass
class MotionCommandWarpCfg(MotionCommandCfg):
    """Configuration for the Warp-consumable motion command."""

    class_type: type = MotionCommandWarp
