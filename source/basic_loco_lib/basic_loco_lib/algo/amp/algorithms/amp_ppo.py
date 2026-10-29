# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications for IsaacNewton: AMPPPO algorithm extending rsl-rl-lib 5.4.1 PPO.

from __future__ import annotations

import torch
import torch.nn as nn

from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from basic_loco_lib.algo.amp.discriminator import Discriminator
from basic_loco_lib.algo.amp.storage import ReplayBuffer
from basic_loco_lib.algo.amp.motion_loader import AMPLoader
from basic_loco_lib.algo.amp.normalizer import Normalizer


class AMPPPO(PPO):
    """PPO with AMP (Adversarial Motion Priors) discriminator — extends rsl-rl-lib 5.4.1 PPO.

    Adds:
    - A discriminator that distinguishes policy transitions from expert motion.
    - A replay buffer of (state, next_state) pairs for the policy.
    - Joint optimisation of actor/critic and discriminator.
    - ``act()`` snapshots the AMP observation from the TensorDict.
    - ``process_env_step()`` inserts the (prev_amp_obs, next_amp_obs) pair into the
      replay buffer (the AMP reward itself is injected by the runner into the reward
      tensor before this call).
    - ``update()`` adds discriminator loss + gradient penalty.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: Discriminator,
        amp_data: AMPLoader,
        amp_normalizer: Normalizer,
        amp_replay_buffer: ReplayBuffer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        device: str = "cpu",
        min_std: list[float] | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Store AMP components before super().__init__
        self.discriminator = discriminator.to(device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.amp_replay_buffer = amp_replay_buffer
        self.min_std = torch.tensor(min_std, device=device) if min_std is not None else None
        self.amp_transition = RolloutStorage.Transition()

        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            use_mixed_precision=use_mixed_precision,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # Extend the configured PPO optimizer with discriminator parameters.
        self.optimizer.add_param_group(
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4}
        )
        self.optimizer.add_param_group(
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2}
        )
        self._clamp_std()

    @torch.no_grad()
    def _clamp_std(self) -> None:
        """Clamp the learned Gaussian parameters, not the cached distribution."""
        if self.min_std is not None:
            distribution = self._raw_actor.distribution
            if distribution.std_type == "scalar":
                distribution.std_param.clamp_(min=self.min_std)
            else:
                distribution.log_std_param.clamp_(min=self.min_std.log())

    # ------------------------------------------------------------------
    # Act — snapshot AMP observation for the upcoming transition pair
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.amp_transition.observations = obs["amp"].clone()
        return super().act(obs)

    # ------------------------------------------------------------------
    # process_env_step — insert AMP replay & delegate to parent
    # ------------------------------------------------------------------

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and insert AMP transition into replay buffer."""
        # Let base PPO store the PPO transition (with AMP-injected rewards)
        super().process_env_step(obs, rewards, dones, extras)

        self.amp_replay_buffer.insert(
            self.amp_transition.observations, self.amp_transition.next_observations
        )
        self.amp_transition.clear()
        self.amp_transition.next_observations = None

    # ------------------------------------------------------------------
    # Update — joint PPO + discriminator training
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run PPO + discriminator optimisation epochs."""
        metric_sums = torch.zeros(7 + int(self.rnd is not None) + int(self.symmetry is not None), device=self.device)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        amp_policy_gen = self.amp_replay_buffer.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches, batch_size
        )
        amp_expert_gen = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches, batch_size
        )

        for batch, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_gen, amp_expert_gen):
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # Symmetry augmentation
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            dist_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Adaptive KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, dist_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Symmetry
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss += self.symmetry.mirror_loss_coeff * symmetry_loss

            # RND
            if self.rnd:
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size])

            # ---- Discriminator ----
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    self.amp_normalizer.update(torch.cat((policy_state, expert_state), dim=0))
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            amp_loss = 0.5 * (
                nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                + nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            )
            grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)
            loss += amp_loss + grad_pen_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            self._clamp_std()

            metrics = [
                value_loss.detach(),
                surrogate_loss.detach(),
                entropy.mean().detach(),
                amp_loss.detach(),
                grad_pen_loss.detach(),
                policy_d.mean().detach(),
                expert_d.mean().detach(),
            ]
            if self.rnd:
                metrics.append(rnd_loss.detach())
            if self.symmetry:
                metrics.append(symmetry_loss.detach())
            metric_sums += torch.stack(metrics)

        n = self.num_learning_epochs * self.num_mini_batches
        metric_means = (metric_sums / n).cpu().tolist()

        self.storage.clear()

        d = {
            "value": metric_means[0],
            "surrogate": metric_means[1],
            "entropy": metric_means[2],
            "amp": metric_means[3],
            "amp_grad_pen": metric_means[4],
            "amp_policy_pred": metric_means[5],
            "amp_expert_pred": metric_means[6],
        }
        metric_idx = 7
        if self.rnd:
            d["rnd"] = metric_means[metric_idx]
            metric_idx += 1
        if self.symmetry:
            d["symmetry"] = metric_means[metric_idx]
        return d

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self) -> dict:
        saved = super().save()
        saved["discriminator_state_dict"] = self.discriminator.state_dict()
        saved["amp_normalizer"] = self.amp_normalizer
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if "amp_normalizer" in loaded_dict:
            self.amp_normalizer = loaded_dict["amp_normalizer"]
        self._clamp_std()
        return result

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()
