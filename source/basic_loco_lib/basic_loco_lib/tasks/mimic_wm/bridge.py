"""Capture WM transitions before the task's managers reset the simulator."""

import torch
import warp as wp

from basic_loco_lib.algo.mimic_wm.contracts import MIMIC_WM_STATE_KEY, MIMIC_WM_TRANSITION_KEY
from basic_loco_lib.algo.mimic_wm.geometry import flatten_state, quat_to_mat

from .diff.mdp import DiffMDP


def as_torch(value):
    return value if torch.is_tensor(value) else wp.to_torch(value)


def resolve_ids(env, ids=None, mask=None):
    if ids is None:
        ids = mask
    if ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(ids, slice):
        return torch.arange(env.num_envs, device=env.device)[ids]
    ids = as_torch(ids).to(env.device)
    return ids.nonzero(as_tuple=True)[0] if ids.dtype == torch.bool else ids.long()


class MimicWmBridge:
    def __init__(self, cfg, render_mode=None, **kwargs):
        self._wm_in_step = False
        self._wm_mdp = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._wm_s1 = torch.zeros(self.num_envs, 71, device=self.device)
        self._wm_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._wm_ep_return = torch.zeros(self.num_envs, device=self.device)
        self._wm_ep_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.extras[MIMIC_WM_STATE_KEY] = self._wm_info()

    @property
    def mimic_wm_mdp(self):
        if self._wm_mdp is None:
            self._wm_mdp = DiffMDP(self)
        return self._wm_mdp

    def _wm_state(self):
        command = self.command_manager.get_term("motion")
        data = command.robot.data
        root = command.robot_anchor_body_index
        return (
            as_torch(data.body_pos_w)[:, root] - self.scene.env_origins,
            quat_to_mat(as_torch(data.body_quat_w)[:, root]),
            as_torch(data.body_lin_vel_w)[:, root],
            as_torch(data.body_ang_vel_w)[:, root],
            as_torch(data.joint_pos),
            as_torch(data.joint_vel),
        )

    def _wm_info(self):
        command = self.command_manager.get_term("motion")
        data = command.robot.data
        context = torch.cat((as_torch(data.default_joint_pos), as_torch(data.default_joint_vel)), -1)
        return {
            "s": flatten_state(self._wm_state()).clone(),
            "f": command.time_steps.clone(),
            "xi": context.clone(),
            "a_prev": as_torch(self.action_manager.action).clone(),
            "ep_step": self.episode_length_buf.clone(),
        }

    def _capture_wm_terminal(self, ids=None, mask=None):
        if not self._wm_in_step:
            return
        ids = resolve_ids(self, ids, mask)
        if ids.numel():
            # A command resample can follow an environment reset in the same step.
            ids = ids[~self._wm_done[ids]]
            self._wm_s1[ids] = flatten_state(self._wm_state())[ids]
            self._wm_done[ids] = True

    def step(self, action):
        pre = self._wm_info()
        term = self.action_manager.get_term("JointPositionAction")
        noise = torch.randn_like(action) * self.cfg.action_noise
        term.execution_noise.copy_(noise)
        self._wm_done.zero_()
        self._wm_in_step = True
        try:
            result = super().step(action)
        finally:
            self._wm_in_step = False
        done = self._wm_done.clone()
        s1 = torch.where(done[:, None], self._wm_s1, flatten_state(self._wm_state()))
        reward = as_torch(self.reward_buf).clone()
        self._wm_ep_return += reward
        self._wm_ep_age += 1
        transition = {
            **pre,
            "s1": s1.clone(),
            "a": (action + noise).clone(),
            "r": reward,
            "done": done,
            "term": done & as_torch(self.reset_terminated),
        }
        if done.any():
            transition["ep_len"] = self._wm_ep_age[done].float().clone()
            transition["ep_ret"] = self._wm_ep_return[done].clone()
            transition["ep_ret_norm"] = (
                self._wm_ep_return[done] / self._wm_ep_age[done] / self.mimic_wm_mdp.max_step_reward
            ).clone()
        self._wm_ep_age[done] = 0
        self._wm_ep_return[done] = 0
        self.extras[MIMIC_WM_TRANSITION_KEY] = transition
        self.extras[MIMIC_WM_STATE_KEY] = self._wm_info()
        return result

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        if hasattr(self, "_wm_done"):
            self._wm_done.zero_()
            self._wm_ep_age.zero_()
            self._wm_ep_return.zero_()
            self.extras[MIMIC_WM_STATE_KEY] = self._wm_info()
        return result
