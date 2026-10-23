# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point key."
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Motion NPZ for mimic, or a motion directory for AMP.",
)
parser.add_argument(
    "--export_only", action="store_true", default=False, help="Export the policy (JIT/ONNX) and exit without playing."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer="none")  # headless by default; basic_loco_lib.sh -p overrides to kit
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

installed_version = metadata.version("rsl-rl-lib")

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import resolve_task_config

import basic_loco_lib.tasks  # noqa: F401


def make_deploy_ready_onnx(env, onnx_path: str):
    """Re-save the exported ONNX as ONE self-contained file with a simple input.

    Two changes over the raw rsl_rl export:
      * weights are merged back from the ``.data`` sidecar -> a single file;
      * the input becomes a plain stack of obs frames, oldest first::

            [frame(t-H+1) | ... | frame(t)]    frame = all obs terms, one step

        which is exactly what the sim2sim scripts and the C++ deployer build.
        IsaacLab instead feeds the history grouped per obs term (all H frames
        of term 0, then term 1, ...); the fixed permutation between the two
        layouts is baked in as a single Gather node, with history length and
        term dims read from the live env. The single input is named ``obs_history``, including for direct
        environments and policies without history.
    """
    from math import prod

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    # obs layout from the live env: group-level history (the manager pushes it
    # down to every term) + per-frame dim of each policy obs term
    history, term_dims = 1, []
    if obs_man := getattr(env.unwrapped, "observation_manager", None):  # None for Direct envs
        policy_cfg = getattr(env.unwrapped.cfg.observations, "policy", None)
        policy_group = "policy"
        if policy_cfg is None:
            policy_cfg = env.unwrapped.cfg.observations.actor
            policy_group = "actor"
        history = int(policy_cfg.history_length or 0)
        if history <= 1:
            # Warp tasks: the experimental ObservationManager does not support
            # group-level history_length ("History reshaping is not implemented
            # yet for warp"); history is implemented per-term by the
            # mdp_warp.make_history_obs wrappers (same blockwise layout), which
            # expose their H as a function attribute.
            hs = {getattr(t.func, "history_length", 1) for t in vars(policy_cfg).values() if hasattr(t, "func")}
            if len(hs) > 1:
                raise ValueError(
                    f"mixed per-term history lengths {sorted(hs)} in the policy group; "
                    "the deploy-side frame-stacked layout requires a uniform H."
                )
            history = int(hs.pop()) if hs else 1
        if history > 1:
            term_dims = [prod(d) // history for d in obs_man.group_obs_term_dim[policy_group]]

    model = onnx.load(onnx_path)  # merges any .data sidecar into memory
    graph = model.graph
    input_name = graph.input[0].name
    if history > 1:
        frame, bounds = sum(term_dims), np.cumsum([0] + term_dims)
        # Gather indices: frame-stacked input -> the per-term order the actor
        # was trained on (term j occupies an (H, dim_j) block)
        perm = np.concatenate(
            [(np.arange(history)[:, None] * frame + np.arange(s, s + d)).ravel() for s, d in zip(bounds[:-1], term_dims)]
        )
        # Expose frame-stacked history and route it through the history permutation.
        graph.input[0].name = "obs_history"
        graph.initializer.append(numpy_helper.from_array(perm, name="history_perm"))
        policy_input_name = f"{input_name}_reordered"
        for node in graph.node:
            for index, node_input in enumerate(node.input):
                if node_input == input_name:
                    node.input[index] = policy_input_name
        graph.node.insert(0, helper.make_node("Gather", ["obs_history", "history_perm"], [policy_input_name], axis=1))
    else:
        graph.input[0].name = "obs_history"
        for node in graph.node:
            for index, node_input in enumerate(node.input):
                if node_input == input_name:
                    node.input[index] = "obs_history"

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)  # single file: weights embedded, no sidecar
    if os.path.isfile(onnx_path + ".data"):
        os.remove(onnx_path + ".data")
    layout = f"obs_history = {history} stacked frames (term dims {term_dims})" if history > 1 else "obs_history = single frame"
    print(f"[INFO] exported {os.path.basename(onnx_path)}: single file, {layout}")


def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    # handle deprecated configurations for rsl-rl version compatibility
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Override the mimic motion file or AMP motion directory.
    if args_cli.motion_file is not None:
        if hasattr(agent_cfg, "amp_motion_files"):
            from basic_loco_lib.tasks.amp.motion_paths import set_motion_directory

            set_motion_directory(env_cfg, agent_cfg, args_cli.motion_file)
            print(f"[INFO] Overriding AMP motion directory to: {agent_cfg.amp_motion_files}")
        elif hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            print(f"[INFO] Overriding motion_file to: {args_cli.motion_file}")
        else:
            print(
                "[WARN] --motion_file applies only to mimic and AMP tasks. Ignoring."
            )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if getattr(agent_cfg, "amp_reward_coef", 0.0) > 0:
        from basic_loco_lib.algo.amp import AmpOnPolicyRunner

        agent_cfg.amp_replay_buffer_size = 1
        runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # make the export deploy-ready: single file, frame-stacked obs history input
    make_deploy_ready_onnx(env, os.path.join(export_model_dir, "policy.onnx"))

    if args_cli.export_only:
        print("[INFO] --export_only: exports done, skipping the play loop")
        env.close()
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    try:
        while True:
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        env.close()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    # GA removed the hydra_task_config decorator: resolve (env_cfg, agent_cfg) explicitly and pass them in.
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, overrides=hydra_args)
    main(env_cfg, agent_cfg)
    simulation_app.close()
