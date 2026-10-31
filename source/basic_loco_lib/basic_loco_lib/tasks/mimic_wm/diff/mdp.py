"""Differentiable counterparts of the original mimic observations and rewards.

Observations match mimic. Rewards use the SuperMimic differentiable tracking
objective; acceleration, torque and contact penalties are intentionally omitted.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import torch

from basic_loco_lib.algo.mimic_wm import geometry as geo
from basic_loco_lib.algo.mimic_wm.contracts import State
from .kinematics import G1Kinematics

NUM_J = 29
OBS_DIM = 154
XI_DIM = 58  # Actual default joint position/velocity; WM-only conditioning.
ACTOR_DIMS = (58, 6, 3, 29, 29, 29)
CRITIC_DIMS = (58, 3, 6, 42, 84, 3, 3, 29, 29, 29)


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9


class MimicReference:
    """Use the original command's loaded frames without resampling the clip."""
    def __init__(self, command):
        motion = command.motion
        self.q, self.qd = motion.joint_pos.clone(), motion.joint_vel.clone()
        self.p = motion.body_pos_w.clone()
        self.R = geo.quat_to_mat(motion.body_quat_w)
        self.v, self.w = motion.body_lin_vel_w.clone(), motion.body_ang_vel_w.clone()
        self.num_frames = len(self.q)

    def ref_state(self, frames):
        return self.p[frames, 0], self.R[frames, 0], self.v[frames, 0], self.w[frames, 0], self.q[frames], self.qd[frames]

    def body_state(self, frames):
        return self.p[frames], self.R[frames], self.v[frames], self.w[frames]


class DiffMDP:
    num_joints = NUM_J
    context_dim = XI_DIM
    obs_dim = OBS_DIM
    critic_dim = sum(CRITIC_DIMS)
    actor_dims = ACTOR_DIMS
    critic_dims = CRITIC_DIMS

    def __init__(self, env):
        command = env.command_manager.get_term('motion')
        robot = command.robot
        self.kin = G1Kinematics(device=env.device, joint_names=robot.joint_names, tracked=command.cfg.body_names)
        if command.cfg.anchor_body_name != 'pelvis' or command.cfg.body_names[0] != 'pelvis':
            raise ValueError('mimic-wm currently requires pelvis as the first tracked body and anchor')
        if len(command.cfg.body_names) != 14:
            raise ValueError('mimic-wm expects the original 14 tracked mimic bodies')
        self.motion = MimicReference(command)
        self.zero_history = "Warp" in type(env).__name__
        self.dt = env.step_dt
        self.default_q = torch.zeros(NUM_J, device=env.device)
        import warp as wp
        self.soft_lo = wp.to_torch(robot.data.soft_joint_pos_limits)[0, :, 0].clone()
        self.soft_hi = wp.to_torch(robot.data.soft_joint_pos_limits)[0, :, 1].clone()
        self.cfg = RewardCfg()
        self.terminations = copy.deepcopy(env.cfg.terminations)
        self.ee_ids = [command.cfg.body_names.index(n) for n in self.terminations.ee_body_pos.params['body_names']]
        self.obs_noise = torch.cat([torch.zeros(58), torch.full((6,), .05), torch.full((3,), .2),
                                    torch.full((29,), .01), torch.full((29,), .5), torch.zeros(29)]).to(env.device)
        self.max_step_reward = 5.0 * self.dt
        middle = (self.kin.q_lo+self.kin.q_hi)/2
        half = (self.kin.q_hi-self.kin.q_lo)*self.cfg.soft_limit_factor/2
        self.soft_lo, self.soft_hi = middle-half, middle+half

    tracking_names = ('motion_global_anchor_pos', 'motion_global_anchor_ori', 'motion_body_pos',
                      'motion_body_ori', 'motion_body_lin_vel', 'motion_body_ang_vel')

    def to(self, device):
        new = copy.copy(self)
        new.kin = self.kin.to(device)
        new.motion = copy.copy(self.motion)
        for name, value in vars(self.motion).items():
            if torch.is_tensor(value): setattr(new.motion, name, value.to(device))
        for name in ('default_q', 'soft_lo', 'soft_hi', 'obs_noise'):
            setattr(new, name, getattr(self, name).to(device))
        return new

    @staticmethod
    def pack_history(history, dims):
        """Original observation manager layout: term, then oldest-to-newest frame."""
        return torch.cat([part.flatten(-2) for part in history.split(dims, -1)], -1)

    @staticmethod
    def unpack_history(obs, dims, hist):
        return torch.cat([part.reshape(*obs.shape[:-1], hist, dim)
                          for part, dim in zip(obs.split([d * hist for d in dims], -1), dims)], -1)

    def obs(self, state, previous_action, context, frames):
        _, rotation, _, angular_velocity, q, qd = state
        ref = self.motion
        relative_R = geo.matmul(rotation.transpose(-1, -2), ref.R[frames, 0])
        return torch.cat((ref.q[frames], ref.qd[frames], geo.mat_to_6d(relative_R),
                          geo.matvec(rotation.transpose(-1, -2), angular_velocity),
                          q-context[..., :29], qd-context[..., 29:58], previous_action), -1)

    def add_noise(self, obs):
        return obs + (2 * torch.rand_like(obs)-1) * self.obs_noise

    def critic_features(self, state, frames, context, previous_action):
        p, R, v, w, q, qd = state
        bp, bR, _, _ = self.kin.forward(*state)
        inv = R.transpose(-1, -2)
        body_p = geo.matvec(inv.unsqueeze(-3), bp-p.unsqueeze(-2)).flatten(-2)
        body_R = geo.mat_to_6d(geo.matmul(inv.unsqueeze(-3), bR)).flatten(-2)
        return torch.cat((self.motion.q[frames], self.motion.qd[frames],
                          geo.matvec(inv, self.motion.p[frames, 0]-p),
                          geo.mat_to_6d(geo.matmul(inv, self.motion.R[frames, 0])), body_p, body_R,
                          geo.matvec(inv,v), geo.matvec(inv,w), q-context[..., :29],
                          qd-context[...,29:58], previous_action), -1)

    def aligned_reference(self, state, frames):
        p, R = state[:2]
        rp, rR, rv, rw = self.motion.body_state(frames)
        delta = geo.yaw_mat(geo.matmul(R, rR[...,0,:,:].transpose(-1,-2)))
        anchor = torch.cat((p[...,:2],rp[...,0,2:3]),-1)
        return (anchor.unsqueeze(-2)+geo.matvec(delta.unsqueeze(-3),rp-rp[...,0:1,:]),
                geo.matmul(delta.unsqueeze(-3),rR),rv,rw)

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        terms: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compare a state with its reference and return the dt-weighted reward."""
        cfg, motion = self.cfg, self.motion
        position, rotation, linear_velocity, angular_velocity, joint_pos, joint_vel = state
        body_p, body_R, body_v, body_w = self.kin.forward(
            position,
            rotation,
            linear_velocity,
            angular_velocity,
            joint_pos,
            joint_vel,
        )
        ref_p, ref_R = motion.ref_state(frames)[:2]
        reference_body_p, reference_body_R, reference_body_v, reference_body_w = motion.body_state(
            frames
        )
        # Reference bodies are re-anchored onto the robot's yaw and xy so the
        # tracking reward is invariant to drift in the ground plane.
        delta_rotation = geo.yaw_mat(geo.matmul(rotation, ref_R.transpose(-1, -2)))
        anchor = torch.cat((position[..., :2], ref_p[..., 2:3]), -1)
        ref_body_p = anchor.unsqueeze(-2) + geo.matvec(
            delta_rotation.unsqueeze(-3), reference_body_p - ref_p.unsqueeze(-2)
        )
        ref_body_R = geo.matmul(delta_rotation.unsqueeze(-3), reference_body_R)
        error_pos = ((ref_body_p - body_p) ** 2).sum(-1).mean(-1)
        error_ori = geo.rot_angle_sq(ref_body_R, body_R).mean(-1)
        error_linvel = ((reference_body_v - body_v) ** 2).sum(-1).mean(-1)
        error_angvel = ((reference_body_w - body_w) ** 2).sum(-1).mean(-1)
        error_root_pos = ((ref_p - position) ** 2).sum(-1)
        error_root_ori = geo.rot_angle_sq(ref_R, rotation)
        tracking = (
            torch.exp(-error_pos / cfg.body_pos_std**2)
            + torch.exp(-error_ori / cfg.body_ori_std**2)
            + torch.exp(-error_linvel / cfg.body_linvel_std**2)
            + torch.exp(-error_angvel / cfg.body_angvel_std**2)
            + 0.5 * torch.exp(-error_root_pos / cfg.root_pos_std**2)
            + 0.5 * torch.exp(-error_root_ori / cfg.root_ori_std**2)
        )
        action_rate = ((action - previous_action) ** 2).sum(-1)
        joint_limit = (
            torch.relu(joint_pos - self.soft_hi) + torch.relu(self.soft_lo - joint_pos)
        ).sum(-1)
        reward = (
            tracking + cfg.w_action_rate * action_rate + cfg.w_joint_limit * joint_limit
        ) * self.dt
        if terms:
            return reward, {
                "track": tracking * self.dt,
                "e_body_pos": error_pos,
                "e_body_ori": error_ori,
                "e_root_pos": error_root_pos,
                "action_rate": action_rate,
                "joint_limit": joint_limit,
            }
        return reward

    def termination_errors(self, state, frames, anchor_state=None):
        bp = self.kin.forward(*state)[0]
        ap = self.aligned_reference(state if anchor_state is None else anchor_state,frames)[0]
        ref = self.motion
        height = (ref.p[frames,0,2]-state[0][...,2]).abs()
        tilt = (ref.R[frames,0,2,2]-state[1][...,2,2]).abs()
        ee = (ap[...,self.ee_ids,2]-bp[...,self.ee_ids,2]).abs().amax(-1)
        return height,tilt,ee

    def termination(self,state,frames,anchor_state=None):
        errors=self.termination_errors(state,frames,anchor_state)
        terms=(self.terminations.anchor_pos,self.terminations.anchor_ori,self.terminations.ee_body_pos)
        return torch.stack([e>t.params['threshold'] for e,t in zip(errors,terms)],-1).any(-1)

    def soft_termination(self,state,frames,anchor_state=None):
        errors=self.termination_errors(state,frames,anchor_state)
        terms=(self.terminations.anchor_pos,self.terminations.anchor_ori,self.terminations.ee_body_pos)
        alive=torch.ones_like(errors[0])
        for e,t,temp in zip(errors,terms,(.02,.04,.02)):
            alive=alive*(1-torch.sigmoid((e-t.params['threshold'])/temp))
        return 1-alive

    def normalize_context(self,context):
        return context

    def valid_imagination_start(self,frames,horizon):
        return frames+horizon<self.motion.num_frames
