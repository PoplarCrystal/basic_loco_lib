# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP on-policy runner extending rsl-rl-lib v5 OnPolicyRunner."""

from __future__ import annotations

import math
import os
import time
import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils.logger import Logger

from basic_loco_lib.algo.amp.algorithms.amp_ppo import AMPPPO
from basic_loco_lib.algo.amp.discriminator import Discriminator
from basic_loco_lib.algo.amp.motion_loader import AMPLoader
from basic_loco_lib.algo.amp.normalizer import Normalizer
from basic_loco_lib.algo.amp.storage import ReplayBuffer


class AmpOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for AMP (Adversarial Motion Priors) extending v5 OnPolicyRunner.

    - Extends the standard learn loop to inject AMP discriminator rewards.
    - Creates the AMPPPO algorithm with discriminator, replay buffer, and expert data loader.
    - Uses pre-reset AMP observations for terminal transitions.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the AMP runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment
        obs = self.env.get_observations()

        # -- Create AMP components --
        # Extract AMP-specific config keys (pop them so they don't flow into actor/critic)
        amp_reward_coef = self.cfg.pop("amp_reward_coef", 0.0)
        amp_motion_files = self.cfg.pop("amp_motion_files", "")
        amp_body_names = self.cfg.pop("amp_body_names", ())
        amp_anchor_name = self.cfg.pop("amp_anchor_name", "")
        amp_discr_hidden_dims = self.cfg.pop("amp_discr_hidden_dims", (1024, 512, 256))
        amp_task_reward_lerp = self.cfg.pop("amp_task_reward_lerp", 0.0)
        min_normalized_std = self.cfg.pop("min_normalized_std", (0.0,))
        amp_replay_buffer_size = self.cfg.pop("amp_replay_buffer_size", 1000000)

        # AMP data loader
        self.amp_data = AMPLoader(
            motion_file=amp_motion_files,
            body_names=list(amp_body_names),
            anchor_name=amp_anchor_name,
            device=self.device,
        )
        print(f"[AMPPPO] AMP observation dimension: {self.amp_data.observation_dim}")

        # Normalizer for AMP observations
        self.amp_normalizer = Normalizer(self.amp_data.observation_dim, device=self.device)

        # Discriminator: input = concat(state, next_state) -> 2 * obs_dim
        self.discriminator = Discriminator(
            self.amp_data.observation_dim * 2,
            amp_reward_coef,
            list(amp_discr_hidden_dims),
            device,
            amp_task_reward_lerp,
        ).to(self.device)

        # Replay buffer for policy AMP transitions
        self.amp_replay_buffer = ReplayBuffer(
            self.amp_data.observation_dim,
            amp_replay_buffer_size,
            self.device,
        )

        # -- Create AMPPPO algorithm --
        self.alg = self._construct_amp_algorithm(obs, list(min_normalized_std))

        # -- Logger --
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    # ------------------------------------------------------------------
    # Algorithm construction (mirrors PPO.construct_algorithm)
    # ------------------------------------------------------------------

    def _construct_amp_algorithm(self, obs, min_std_list: list[float]) -> AMPPPO:
        """Build AMPPPO from config, mirroring :meth:`PPO.construct_algorithm`."""
        # Resolve class callables
        alg_class: type[AMPPPO] = resolve_callable(self.cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(self.cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(self.cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in self.cfg["algorithm"] and self.cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg.get("obs_groups", {}), default_sets)

        # Resolve RND / symmetry configs
        self.cfg["algorithm"] = resolve_rnd_config(self.cfg["algorithm"], obs, self.cfg["obs_groups"], self.env)
        self.cfg["algorithm"] = resolve_symmetry_config(self.cfg["algorithm"], self.env)

        # Create actor and critic models
        actor: MLPModel = actor_class(obs, self.cfg["obs_groups"], "actor", self.env.num_actions, **self.cfg["actor"]).to(
            self.device
        )
        print(f"Actor Model: {actor}")
        if self.cfg["algorithm"].pop("share_cnn_encoders", None):
            self.cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, self.cfg["obs_groups"], "critic", 1, **self.cfg["critic"]).to(self.device)
        print(f"Critic Model: {critic}")

        # Storage
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        # Create AMPPPO
        alg: AMPPPO = alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            discriminator=self.discriminator,
            amp_data=self.amp_data,
            amp_normalizer=self.amp_normalizer,
            amp_replay_buffer=self.amp_replay_buffer,
            device=self.device,
            min_std=min_std_list,
            **self.cfg["algorithm"],
            multi_gpu_cfg=self.cfg.get("multi_gpu"),
        )

        return alg

    # ------------------------------------------------------------------
    # Learning loop — adds AMP reward injection
    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop with AMP discriminator reward."""
        # Randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        check_for_nan = self.cfg.get("check_for_nan", True)
        if check_for_nan:
            self._check_finite(dict(obs.items()), self.current_learning_iteration, 0)

        # Broadcast parameters if distributed
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize logging
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for step in range(1, self.cfg["num_steps_per_env"] + 1):
                    # Sample actions (AMPPPO.act saves amp_obs from obs TensorDict)
                    actions = self.alg.act(obs)
                    if check_for_nan:
                        self._check_finite({"policy_actions": actions}, it, step)

                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # ---- AMP reward computation ----
                    next_amp_obs = torch.where(
                        dones.bool().unsqueeze(-1),
                        extras["terminal_amp_observations"].to(self.device),
                        obs["amp"],
                    )
                    if check_for_nan:
                        self._check_finite(
                            {**dict(obs.items()), "rewards": rewards, "dones": dones, "next_amp_obs": next_amp_obs},
                            it,
                            step,
                        )
                    amp_reward, amp_score = self.alg.discriminator.predict_amp_reward(
                        self.alg.amp_transition.observations,
                        next_amp_obs,
                        rewards,
                        normalizer=self.alg.amp_normalizer,
                    )
                    # predict_amp_reward already returns the task/AMP mixture.
                    rewards = amp_reward.squeeze(-1)
                    if check_for_nan:
                        self._check_finite({"amp_rewards": rewards, "amp_score": amp_score}, it, step)
                    self.alg.amp_transition.next_observations = next_amp_obs

                    # Process the step (stores PPO transition + AMP transition in replay buffer)
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    # Logger bookkeeping
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy + discriminator
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        # Final save and cleanup
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    # ------------------------------------------------------------------
    # Save / Load — include discriminator + normalizer
    # ------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state including discriminator."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state including discriminator."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_finite(self, tensors: dict[str, torch.Tensor], iteration: int, step: int) -> None:
        """Fail before invalid values reach physics, the discriminator, or rollout storage."""
        invalid = torch.stack([~torch.isfinite(tensor).all() for tensor in tensors.values()]).cpu().tolist()
        messages = []
        for (name, tensor), is_invalid in zip(tensors.items(), invalid):
            if not is_invalid:
                continue
            mask = ~torch.isfinite(tensor).reshape(tensor.shape[0], -1)
            env_ids = torch.where(mask.any(dim=1))[0].cpu().tolist()
            message = (
                f"NaN/Inf in '{name}' at learning iteration {iteration}, rollout step {step}"
                f"; env ids (first 32 of {len(env_ids)}): {env_ids[:32]}"
            )
            if self.cfg.get("detailed_nan_check", False) and tensor.ndim >= 2:
                feature_ids = torch.where(mask.any(dim=0))[0].cpu().tolist()
                message += f"; feature ids (first 64 of {len(feature_ids)}): {feature_ids[:64]}"
                manager = self.env.unwrapped.observation_manager
                group = "amp" if name == "next_amp_obs" else name
                if group in manager.active_terms:
                    offset = 0
                    terms = []
                    for term, dims in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
                        end = offset + math.prod(dims)
                        if any(offset <= feature < end for feature in feature_ids):
                            terms.append(term)
                        offset = end
                    message += f"; {group} terms: {terms}"
            messages.append(message)
        if messages:
            raise ValueError("\n".join(messages))
