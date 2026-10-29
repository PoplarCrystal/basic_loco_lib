from __future__ import annotations

import torch

import warp as wp

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand


@wp.kernel
def _init_rng(seed: int, state: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    state[i] = wp.rand_init(seed, i)


@wp.kernel
def _compute_velocity_command(
    time_left: wp.array(dtype=wp.float32),
    command_counter: wp.array(dtype=wp.int64),
    command: wp.array(dtype=wp.float32, ndim=2),
    heading_target: wp.array(dtype=wp.float32),
    heading_env: wp.array(dtype=wp.bool),
    standing_env: wp.array(dtype=wp.bool),
    heading_w: wp.array(dtype=wp.float32),
    rng_state: wp.array(dtype=wp.uint32),
    dt: float,
    resample_min: float,
    resample_max: float,
    lin_x_min: float,
    lin_x_max: float,
    lin_y_min: float,
    lin_y_max: float,
    ang_z_min: float,
    ang_z_max: float,
    target_heading_min: float,
    target_heading_max: float,
    heading_command: bool,
    rel_heading_envs: float,
    rel_standing_envs: float,
    heading_stiffness: float,
):
    i = wp.tid()
    t = time_left[i] - dt
    state = rng_state[i]
    if t <= 0.0:
        t = wp.randf(state, resample_min, resample_max)
        command[i, 0] = wp.randf(state, lin_x_min, lin_x_max)
        command[i, 1] = wp.randf(state, lin_y_min, lin_y_max)
        command[i, 2] = wp.randf(state, ang_z_min, ang_z_max)
        if heading_command:
            heading_target[i] = wp.randf(state, target_heading_min, target_heading_max)
            heading_env[i] = wp.randf(state, 0.0, 1.0) <= rel_heading_envs
        standing_env[i] = wp.randf(state, 0.0, 1.0) <= rel_standing_envs
        command_counter[i] = command_counter[i] + wp.int64(1)
    if heading_command and heading_env[i]:
        pi = 3.141592653589793
        error = heading_target[i] - heading_w[i]
        error = error - 2.0 * pi * wp.floor((error + pi) / (2.0 * pi))
        command[i, 2] = wp.clamp(heading_stiffness * error, ang_z_min, ang_z_max)
    if standing_env[i]:
        command[i, 0] = 0.0
        command[i, 1] = 0.0
        command[i, 2] = 0.0
    time_left[i] = t
    rng_state[i] = state


def _wp(value):
    if isinstance(value, wp.array):
        return value
    if hasattr(value, "warp"):
        return value.warp
    return wp.from_torch(value.contiguous() if not value.is_contiguous() else value)


class WarpUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command update that avoids per-step Torch nonzero indexing."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._rng_state = wp.zeros((self.num_envs,), dtype=wp.uint32, device=self.device)
        wp.launch(
            _init_rng,
            dim=self.num_envs,
            inputs=[int(torch.initial_seed() % (2**31 - 1)), self._rng_state],
            device=self.device,
        )

    def compute(self, dt: float):
        self._update_metrics()
        cfg = self.cfg
        wp.launch(
            _compute_velocity_command,
            dim=self.num_envs,
            inputs=[
                _wp(self.time_left),
                _wp(self.command_counter),
                _wp(self.vel_command_b),
                _wp(self.heading_target),
                _wp(self.is_heading_env),
                _wp(self.is_standing_env),
                _wp(self.robot.data.heading_w),
                self._rng_state,
                dt,
                *cfg.resampling_time_range,
                *cfg.ranges.lin_vel_x,
                *cfg.ranges.lin_vel_y,
                *cfg.ranges.ang_vel_z,
                *(cfg.ranges.heading if cfg.ranges.heading is not None else (0.0, 0.0)),
                cfg.heading_command,
                cfg.rel_heading_envs,
                cfg.rel_standing_envs,
                cfg.heading_control_stiffness,
            ],
            device=self.device,
        )
