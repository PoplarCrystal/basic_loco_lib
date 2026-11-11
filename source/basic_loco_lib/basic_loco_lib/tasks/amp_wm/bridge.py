"""Capture world-model transitions for the AMP locomotion task.

Reads the robot directly (no motion command)
and appends the velocity command to the world-model context so the
differentiable MDP can reconstruct the AMP policy observation and reward in
imagination. The last three context slots hold ``[vx, vy, wz]``.
"""

import torch

from basic_loco_lib.algo.amp_wm.contracts import AMP_WM_STATE_KEY, AMP_WM_TRANSITION_KEY
from basic_loco_lib.algo.amp_wm.geometry import flatten_state, quat_to_mat

from .diff.mdp import AmpDiffMDP


def as_torch(value):
    """Torch view of a data buffer (torch tensor, IsaacLab proxy, or Warp array)."""
    if torch.is_tensor(value):
        return value
    torch_view = getattr(value, "torch", None)
    if torch_view is not None:
        return torch_view
    import warp as wp

    return wp.to_torch(value)


def resolve_ids(env, ids=None, mask=None):
    if ids is None:
        ids = mask
    if ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(ids, slice):
        return torch.arange(env.num_envs, device=env.device)[ids]
    ids = as_torch(ids).to(env.device)
    return ids.nonzero(as_tuple=True)[0] if ids.dtype == torch.bool else ids.long()


class AmpWmBridge:
    command_name = "twist"
    action_term_name = "JointPositionAction"

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._wm_in_step = False
        self._wm_mdp = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self._root_body_index = list(self.robot.body_names).index("pelvis")
        self._context_extra = self._read_context_features()
        self.context_extra_dim = self._context_extra.shape[1]
        self._wm_s1 = torch.zeros(self.num_envs, 71, device=self.device)
        self._wm_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._wm_contact1 = torch.zeros(self.num_envs, 2, device=self.device)
        self._wm_recovery1 = torch.zeros(self.num_envs, 3, device=self.device)
        self._wm_ep_return = torch.zeros(self.num_envs, device=self.device)
        self._wm_ep_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.extras[AMP_WM_STATE_KEY] = self._wm_info()

    @property
    def amp_wm_mdp(self):
        if self._wm_mdp is None:
            self._wm_mdp = AmpDiffMDP(self)
        return self._wm_mdp

    def _wm_state(self):
        data = self.robot.data
        root = self._root_body_index
        return (
            as_torch(data.body_pos_w)[:, root] - self.scene.env_origins,
            quat_to_mat(as_torch(data.body_quat_w)[:, root]),
            as_torch(data.body_lin_vel_w)[:, root],
            as_torch(data.body_ang_vel_w)[:, root],
            as_torch(data.joint_pos),
            as_torch(data.joint_vel),
        )

    def _wm_info(self):
        data = self.robot.data
        command = self.command_manager.get_command(self.command_name)
        context = torch.cat(
            (
                as_torch(data.default_joint_pos),
                as_torch(data.default_joint_vel),
                command,
                self._context_extra,
            ),
            -1,
        )
        return {
            "s": flatten_state(self._wm_state()).clone(),
            "f": self.episode_length_buf.clone(),
            "xi": context.clone(),
            "a_prev": as_torch(self.action_manager.action).clone(),
            "ep_step": self.episode_length_buf.clone(),
            "recovery": self._recovery_state(),
        }

    def _recovery_state(self):
        manager = self.termination_manager
        if not hasattr(manager, "_delay_env_mask"):
            return torch.zeros(self.num_envs, 3, device=self.device)
        enabled = manager._delay_env_mask & (manager._max_delay_steps > 0)
        return torch.stack((
            enabled.float(), manager._delay_counters.float(),
            torch.full_like(manager._delay_counters, manager._max_delay_steps).float(),
        ), -1).clone()

    def _foot_shape_indices(self) -> torch.Tensor:
        names = list(self.robot.body_names)
        body_ids = [
            names.index("left_ankle_roll_link"),
            names.index("right_ankle_roll_link"),
        ]
        backend_ids = self.robot.map_body_ids_to_backend(body_ids)
        if hasattr(backend_ids, "tolist"):
            backend_ids = backend_ids.tolist()
        body_shapes = self.robot._root_view.body_shapes
        shape_ids = [int(shape_id) for body_id in backend_ids for shape_id in body_shapes[body_id]]
        return torch.tensor(shape_ids, dtype=torch.long, device=self.device)

    def _read_friction_features(self) -> torch.Tensor:
        robot = self.robot
        try:
            model = self.sim.physics_manager.get_model()
            mu = as_torch(robot._root_view.get_attribute("shape_material_mu", model)[:, 0]).float()
            restitution = as_torch(
                robot._root_view.get_attribute("shape_material_restitution", model)[:, 0]
            ).float()
            if mu.ndim == 1:
                mu = mu.unsqueeze(0)
            if restitution.ndim == 1:
                restitution = restitution.unsqueeze(0)
            if mu.shape[0] != self.num_envs and mu.shape[-1] == self.num_envs:
                mu = mu.transpose(0, 1)
                restitution = restitution.transpose(0, 1)
            shape_ids = self._foot_shape_indices()
            if mu.ndim == 2 and mu.shape[0] == self.num_envs and mu.shape[1] > int(shape_ids.max()):
                mu = mu[:, shape_ids]
                restitution = restitution[:, shape_ids]
            return torch.cat((mu, restitution), -1)
        except Exception:
            pass
        try:
            props = as_torch(robot.root_view.get_material_properties()).float()
            return props.reshape(self.num_envs, -1)
        except Exception:
            pass
        return torch.zeros(self.num_envs, 0, device=self.device)

    def _read_context_features(self) -> torch.Tensor:
        torso = list(self.robot.body_names).index("torso_link")
        com = as_torch(self.robot.data.body_com_pos_b)[:, torso].float()
        return torch.cat((com, self._read_friction_features()), -1)

    def _contact_state(self) -> torch.Tensor:
        sensors = getattr(self.scene, "sensors", None) or {}
        sensor = sensors.get("feet_ground_contact") if hasattr(sensors, "get") else None
        if sensor is None or sensor.data.net_forces_w is None:
            return torch.zeros(self.num_envs, 2, device=self.device)
        names = list(sensor.body_names)
        ids = [
            names.index("left_ankle_roll_link"),
            names.index("right_ankle_roll_link"),
        ]
        force = as_torch(sensor.data.net_forces_w)
        return (torch.linalg.vector_norm(force[:, ids], dim=-1) > 1.0).float()

    def _capture_wm_terminal(self, ids=None, mask=None):
        if not self._wm_in_step:
            return
        ids = resolve_ids(self, ids, mask)
        if ids.numel():
            # A command resample can follow an environment reset in the same step.
            ids = ids[~self._wm_done[ids]]
            self._wm_s1[ids] = flatten_state(self._wm_state())[ids]
            self._wm_contact1[ids] = self._contact_state()[ids]
            self._wm_recovery1[ids] = self._recovery_state()[ids]
            self._wm_done[ids] = True

    def step(self, action):
        pre = self._wm_info()
        term = self.action_manager.get_term(self.action_term_name)
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
        episode_done = as_torch(result[2]) | as_torch(result[3])
        post = self._wm_info()
        self._wm_ep_return += reward
        self._wm_ep_age += 1
        transition = {
            **pre,
            "s1": s1.clone(),
            "a": (action + noise).clone(),
            "r": reward,
            "done": done,
            "episode_done": episode_done.clone(),
            "term": as_torch(result[2]).clone(),
            "contact": torch.where(done[:, None], self._wm_contact1, self._contact_state()),
            "a_clean": action.clone(),
            "xi1": torch.where(done[:, None], pre["xi"], post["xi"]),
            "recovery1": torch.where(done[:, None], self._wm_recovery1, post["recovery"]),
        }
        if episode_done.any():
            transition["ep_len"] = self._wm_ep_age[episode_done].float().clone()
            transition["ep_ret"] = self._wm_ep_return[episode_done].clone()
            transition["ep_ret_norm"] = (
                self._wm_ep_return[episode_done]
                / self._wm_ep_age[episode_done]
                / self.amp_wm_mdp.max_step_reward
            ).clone()
        self._wm_ep_age[episode_done] = 0
        self._wm_ep_return[episode_done] = 0
        self.extras[AMP_WM_TRANSITION_KEY] = transition
        self.extras[AMP_WM_STATE_KEY] = post
        return result

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        if hasattr(self, "_wm_done"):
            self._wm_done.zero_()
            self._wm_ep_age.zero_()
            self._wm_ep_return.zero_()
            self.extras[AMP_WM_STATE_KEY] = self._wm_info()
        return result


__all__ = ["AmpWmBridge"]
