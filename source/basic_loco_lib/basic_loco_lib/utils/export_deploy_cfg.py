import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def _write_deploy_yaml(cfg: dict, log_dir: str):
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)


def _collect_common_cfg(env):
    asset: Articulation = env.scene["robot"]
    joint_sdk_names = getattr(env.cfg.scene.robot, "joint_sdk_names", asset.joint_names)
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness.torch[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping.torch[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos.torch[0].detach().cpu().numpy().tolist()
    return cfg, asset


def _collect_command_cfg(env, cfg: dict):
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges


def _scale_to_list(scale, obs_dim: int):
    if scale is None:
        return [1.0 for _ in range(obs_dim)]
    scale = scale.detach().cpu().numpy().tolist()
    if isinstance(scale, float):
        return [scale for _ in range(obs_dim)]
    return scale


def _export_deploy_cfg_direct(env, log_dir):
    cfg, _asset = _collect_common_cfg(env)
    _collect_command_cfg(env, cfg)

    # --- actions ---
    action_dim = int(env.cfg.action_space)
    cfg["actions"] = {
        "JointPositionAction": {
            "clip": list(env.cfg.action_clip) if env.cfg.action_clip is not None else None,
            "joint_names": [".*"],
            "scale": [float(env.cfg.action_scale) for _ in range(action_dim)],
            "offset": env._default_joint_pos[0].detach().cpu().numpy().tolist(),
            "joint_ids": None,
        }
    }

    # --- observations ---
    cfg["observations"] = {}
    for obs_name, func, params, scale, _noise_cfg, _modifier_cfgs, obs_cfg in env._policy_obs_terms:
        obs_dims = tuple(func(env, **params).shape)
        term_cfg = obs_cfg.copy()
        term_cfg.scale = _scale_to_list(scale, obs_dims[1])
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if not term_cfg.history_length:
            term_cfg.history_length = 1

        term_cfg = term_cfg.to_dict()
        for key in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[key]
        cfg["observations"][obs_name] = term_cfg

    _write_deploy_yaml(cfg, log_dir)


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    # ``export_deploy_cfg`` introspects ActionManager / ObservationManager
    # to dump a sim2real-ready yaml.  Direct-workflow envs keep equivalent
    # metadata in their own cached observation/action tables instead of full
    # Manager internals, so they use a separate dumper below.
    if not isinstance(env, ManagerBasedRLEnv):
        if all(hasattr(env, attr) for attr in ("_policy_obs_terms", "_default_joint_pos")):
            _export_deploy_cfg_direct(env, log_dir)
            return
        print(
            "[INFO] export_deploy_cfg: skipping - not a ManagerBasedRLEnv "
            f"(got {type(env).__name__}); deploy.yaml will not be created."
        )
        return

    cfg, asset = _collect_common_cfg(env)

    # --- commands ---
    _collect_command_cfg(env, cfg)

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if hasattr(term_cfg, "use_default_offset"):
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    policy_group = "policy" if "policy" in env.observation_manager.active_terms else "actor"
    obs_names = env.observation_manager.active_terms[policy_group]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs[policy_group]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            del term_cfg[_]
        cfg["observations"][obs_name] = term_cfg

    # --- save config file ---
    _write_deploy_yaml(cfg, log_dir)
