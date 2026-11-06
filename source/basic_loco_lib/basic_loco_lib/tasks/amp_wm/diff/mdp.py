"""Differentiable AMP observation/reward heads computed from the world-model state.

The AMP policy observation, privileged critic observation and discriminator
features are all analytic functions of the 71-D rigid-body state plus the
velocity command. Forward kinematics (:class:`G1Kinematics`) makes them
differentiable, so the policy can be optimised through the learned world
model.

Body velocities need one convention note: IsaacLab exposes
``body_lin_vel_w`` as a *center-of-mass* velocity, while the AMP expert NPZ
files (and the Warp AMP observation kernels) use link-origin velocities. The
world-model state follows the IsaacLab convention, so the root COM velocity is
converted to a link velocity before the FK recursion.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from basic_loco_lib.algo.amp_wm import geometry as geo
from basic_loco_lib.algo.amp_wm.contracts import State
from .kinematics import G1Kinematics

NUM_J = 29

# Per-frame layouts, mirroring the AMP Warp task's ``policy`` / ``critic`` groups.
ACTOR_DIMS = (3, 3, 3, 29, 29, 29)
CRITIC_DIMS = (3, 3, 3, 3, 29, 29, 29, 39, 78)
OBS_DIM = sum(ACTOR_DIMS)
CRITIC_DIM = sum(CRITIC_DIMS)

# World-model context: default joint position / velocity (58) + velocity command (3).
XI_DIM = 58 + 3

# AMP discriminator bodies. ``torso_link`` is the anchor used for relative
# body positions / orientations and is appended to the FK query set.
AMP_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
ANCHOR_NAME = "torso_link"


@dataclass
class RewardCfg:
    """Differentiable counterpart of the AMP task's reward terms."""

    lin_vel_std: float = 1.0
    ang_vel_std: float = 3.14
    height_std: float = 0.3
    body_ang_vel_std: float = 3.14
    w_body_ang_vel_xy: float = 0.5
    w_action_rate: float = -0.01
    # ~20x the real-env ``joint_acc_l2`` weight (-2.5e-7): the WM uses the
    # control-step average acceleration (50 Hz), whose squared magnitude is
    # measured at ~7x (noiseless) to ~34x (with execution noise) smaller than
    # the sim's 200 Hz instantaneous acceleration.
    w_joint_acc: float = -1e-5
    w_joint_limit: float = -2.0


def as_torch(value):
    if torch.is_tensor(value):
        return value
    import warp as wp

    return wp.to_torch(value)


class AmpDiffMDP:
    """Task-owned differentiable MDP for AMP locomotion in imagination."""

    num_joints = NUM_J
    context_dim = XI_DIM
    obs_dim = OBS_DIM
    critic_dim = CRITIC_DIM
    actor_dims = ACTOR_DIMS
    critic_dims = CRITIC_DIMS

    amp_body_names = AMP_BODY_NAMES
    amp_anchor_name = ANCHOR_NAME
    amp_feature_dim = 15 * len(AMP_BODY_NAMES)

    # Value-function supremum used to rescale the learner's clip bounds. The
    # task reward is dt-scaled like the manager's reward; the AMP style reward
    # is not (matching ``AmpOnPolicyRunner``'s reward mixing).
    max_step_reward = 0.1

    def __init__(self, env):
        robot = env.scene["robot"]
        self.kin = G1Kinematics(
            device=env.device,
            joint_names=robot.joint_names,
            tracked=AMP_BODY_NAMES + (ANCHOR_NAME,),
        )
        self.anchor_idx = len(AMP_BODY_NAMES)
        self.root_body_index = self.kin.tracked_idx[0]
        self.anchor_body_index = self.kin.tracked_idx[self.anchor_idx]
        self.body_count = len(AMP_BODY_NAMES)
        self.dt = env.step_dt

        soft_limits = as_torch(robot.data.soft_joint_pos_limits)
        self.soft_lo = soft_limits[0, :, 0].clone()
        self.soft_hi = soft_limits[0, :, 1].clone()
        self.default_height = float(robot.cfg.init_state.pos[2])

        terminations = env.cfg.terminations
        self.height_threshold = float(terminations.bad_root_height.params["minimum_height"])
        self.tilt_threshold = float(terminations.bad_orientation.params["limit_angle"])

        self.cfg = RewardCfg()
        # The Warp observation manager zero-pads history before the first frames.
        self.zero_history = "Warp" in type(env).__name__
        # Per-frame policy corruption, matching the AMP observation terms.
        self.obs_noise = torch.cat(
            (
                torch.zeros(3),  # command
                torch.full((3,), 0.05),  # projected gravity
                torch.full((3,), 0.2),  # base angular velocity
                torch.full((29,), 0.01),  # joint positions
                torch.full((29,), 0.5),  # joint velocities
                torch.zeros(29),  # last action
            )
        ).to(env.device)

    # ------------------------------------------------------------------
    # Differentiable kinematics
    # ------------------------------------------------------------------

    def _fk(self, state: State):
        """Run FK from a state whose root velocity is a COM velocity."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        com_offset = self.kin.com_pos[self.root_body_index]
        link_velocity = lin_vel + torch.cross(
            ang_vel,
            geo.matvec(rotation, -com_offset),
            dim=-1,
        )
        return self.kin.forward(
            position,
            rotation,
            link_velocity,
            ang_vel,
            joint_pos,
            joint_vel,
            all_bodies=False,
        )

    def amp_features(self, state: State) -> torch.Tensor:
        """Discriminator features, in the AMP expert loader's layout."""
        body_pos, body_rot, body_lin, body_ang = self._fk(state)
        anchor = self.anchor_idx
        anchors = body_pos[..., anchor, :]
        inv_anchor = body_rot[..., anchor, :, :].transpose(-1, -2)

        relative_pos = geo.matvec(
            inv_anchor.unsqueeze(-3), body_pos - anchors.unsqueeze(-2)
        )
        relative_rot = geo.mat_to_6d(
            geo.matmul(inv_anchor.unsqueeze(-3), body_rot)
        )
        lin_vel_b = geo.matvec(
            body_rot[..., : self.body_count, :, :].transpose(-1, -2),
            body_lin[..., : self.body_count, :],
        )
        ang_vel_b = geo.matvec(
            body_rot[..., : self.body_count, :, :].transpose(-1, -2),
            body_ang[..., : self.body_count, :],
        )
        return torch.cat(
            (
                relative_pos[..., : self.body_count, :].flatten(-2),
                relative_rot[..., : self.body_count, :].flatten(-2),
                lin_vel_b.flatten(-2),
                ang_vel_b.flatten(-2),
            ),
            -1,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    @staticmethod
    def pack_history(history: torch.Tensor, dims) -> torch.Tensor:
        """Original observation manager layout: term, then oldest-to-newest frame."""
        return torch.cat([part.flatten(-2) for part in history.split(dims, -1)], -1)

    @staticmethod
    def unpack_history(obs: torch.Tensor, dims, hist: int) -> torch.Tensor:
        return torch.cat(
            [
                part.reshape(*obs.shape[:-1], hist, dim)
                for part, dim in zip(obs.split([d * hist for d in dims], -1), dims)
            ],
            -1,
        )

    def obs(self, state: State, previous_action, context, frames) -> torch.Tensor:
        _, rotation, _, ang_vel, joint_pos, joint_vel = state
        command = context[..., 58:61]
        gravity_b = -rotation.transpose(-1, -2)[..., :, 2]
        ang_vel_b = geo.matvec(rotation.transpose(-1, -2), ang_vel)
        return torch.cat(
            (
                command,
                gravity_b,
                ang_vel_b,
                joint_pos - context[..., :29],
                joint_vel - context[..., 29:58],
                previous_action,
            ),
            -1,
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

    def critic_features(self, state: State, frames, context, previous_action) -> torch.Tensor:
        _, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        command = context[..., 58:61]
        gravity_b = -rotation.transpose(-1, -2)[..., :, 2]
        base_lin_vel_b = geo.matvec(rotation.transpose(-1, -2), lin_vel)
        base_ang_vel_b = geo.matvec(rotation.transpose(-1, -2), ang_vel)

        body_pos, body_rot, _, _ = self._fk(state)
        anchor = self.anchor_idx
        anchors = body_pos[..., anchor, :]
        inv_anchor = body_rot[..., anchor, :, :].transpose(-1, -2)
        body_pos_b = geo.matvec(
            inv_anchor.unsqueeze(-3), body_pos - anchors.unsqueeze(-2)
        )[..., : self.body_count, :].flatten(-2)
        body_ori_b = geo.mat_to_6d(
            geo.matmul(inv_anchor.unsqueeze(-3), body_rot)
        )[..., : self.body_count, :].flatten(-2)

        return torch.cat(
            (
                command,
                gravity_b,
                base_lin_vel_b,
                base_ang_vel_b,
                joint_pos - context[..., :29],
                joint_vel - context[..., 29:58],
                previous_action,
                body_pos_b,
                body_ori_b,
            ),
            -1,
        )

    # ------------------------------------------------------------------
    # Reward and soft termination
    # ------------------------------------------------------------------

    def reward(
        self,
        state: State,
        frames,
        action,
        previous_action,
        context=None,
        prev_state: State | None = None,
        terms: bool = False,
    ):
        """Velocity-tracking / regularization reward mirroring the AMP task."""
        if context is None:
            raise ValueError(
                "AmpDiffMDP.reward requires the world-model context (velocity command)"
            )
        command = context[..., 58:61]
        position, rotation, _, ang_vel, joint_pos, _ = state
        _, body_rot, body_lin, body_ang = self._fk(state)
        anchor = self.anchor_idx
        anchor_rot = body_rot[..., anchor, :, :]
        anchor_ang = body_ang[..., anchor, :]
        # The AMP task tracks the anchor's COM velocity (``body_lin_vel_w``).
        anchor_lin = body_lin[..., anchor, :] + torch.cross(
            anchor_ang,
            geo.matvec(anchor_rot, self.kin.com_pos[self.anchor_body_index]),
            dim=-1,
        )

        cos_yaw = anchor_rot[..., 0, 0]
        sin_yaw = anchor_rot[..., 1, 0]
        norm = torch.sqrt(cos_yaw * cos_yaw + sin_yaw * sin_yaw + 1e-9)
        cos_yaw, sin_yaw = cos_yaw / norm, sin_yaw / norm
        command_world = torch.stack(
            (
                cos_yaw * command[..., 0] - sin_yaw * command[..., 1],
                sin_yaw * command[..., 0] + cos_yaw * command[..., 1],
                torch.zeros_like(cos_yaw),
            ),
            -1,
        )
        lin_error = (command_world - anchor_lin).square().sum(-1)
        lin_reward = torch.exp(-lin_error / self.cfg.lin_vel_std**2)

        anchor_ang_b = geo.matvec(anchor_rot.transpose(-1, -2), anchor_ang)
        ang_error = (command[..., 2] - anchor_ang[..., 2]).square() + (
            anchor_ang_b[..., 0].square() + anchor_ang_b[..., 1].square()
        )
        ang_reward = torch.exp(-ang_error / self.cfg.ang_vel_std**2)

        height_error = position[..., 2] - self.default_height
        height_reward = torch.exp(-height_error.square() / self.cfg.height_std**2)

        pelvis_ang_b = geo.matvec(rotation.transpose(-1, -2), ang_vel)
        body_reward = torch.exp(
            -(pelvis_ang_b[..., 0].square() + pelvis_ang_b[..., 1].square())
            / self.cfg.body_ang_vel_std**2
        )

        action_rate = (action - previous_action).square().sum(-1)
        joint_limit = (
            torch.relu(joint_pos - self.soft_hi) + torch.relu(self.soft_lo - joint_pos)
        ).sum(-1)
        # Control-step average joint acceleration: joint_vel is part of the WM
        # state, so the gradient flows through the frozen WM to the action.
        if prev_state is not None:
            joint_acc = ((state[5] - prev_state[5]) / self.dt).square().sum(-1)
        else:
            joint_acc = torch.zeros_like(joint_limit)

        reward = (
            lin_reward
            + ang_reward
            + height_reward
            + self.cfg.w_body_ang_vel_xy * body_reward
            + self.cfg.w_action_rate * action_rate
            + self.cfg.w_joint_acc * joint_acc
            + self.cfg.w_joint_limit * joint_limit
        ) * self.dt
        if terms:
            return reward, {
                "track_lin": lin_reward,
                "track_ang": ang_reward,
                "track_height": height_reward,
                "body_ang_vel_xy": body_reward,
                "action_rate": action_rate,
                "joint_acc": joint_acc,
                "joint_limit": joint_limit,
            }
        return reward

    def soft_termination(self, state: State, frames, anchor_state=None) -> torch.Tensor:
        position, rotation = state[0], state[1]
        height_error = self.height_threshold - position[..., 2]
        cosine = rotation[..., 2, 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        tilt = torch.acos(cosine)
        alive = (1.0 - torch.sigmoid(height_error / 0.02)) * (
            1.0 - torch.sigmoid((tilt - self.tilt_threshold) / 0.04)
        )
        return 1.0 - alive

    # ------------------------------------------------------------------
    # Protocol plumbing
    # ------------------------------------------------------------------

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return context

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        # No reference clip: every replay row can start an imagination rollout.
        return torch.ones_like(frames, dtype=torch.bool)

    def to(self, device: str | torch.device):
        new = copy.copy(self)
        new.kin = self.kin.to(device)
        for name in ("soft_lo", "soft_hi", "obs_noise"):
            setattr(new, name, getattr(self, name).to(device))
        return new


__all__ = [
    "ACTOR_DIMS",
    "AMP_BODY_NAMES",
    "ANCHOR_NAME",
    "AmpDiffMDP",
    "CRITIC_DIMS",
    "RewardCfg",
]
