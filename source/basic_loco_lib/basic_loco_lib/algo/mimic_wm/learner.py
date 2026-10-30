"""Train mimic's original observation policy through learned dynamics.

The collect/WM-fit/imagination schedule follows Uni-SuperMimic. Execution noise
is a WM training input, but never replaces the clean policy action in history.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .contracts import (
    MIMIC_WM_STATE_KEY,
    MIMIC_WM_TRANSITION_KEY,
    State,
    get_mimic_wm_mdp,
)
from . import networks
from .geometry import unflatten_state
from .networks import Critic, Policy, WorldModel, mlp
from .replay import ReplayBuffer


@dataclass
class MimicWmConfig:
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    hist: int = 4
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    policy_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    clip_value_target: bool = True
    # Value-clip bounds expressed as multiples of the task's value-function
    # supremum ``max_step_reward / (1 - gamma)``; the learner rescales them in
    # ``__init__`` once the task MDP is known. The defaults reproduce
    # Uni-SuperMimic's raw (-100, 10) for the G1 task (supremum 10): the max
    # side is the supremum itself, the min side only needs to sit below any
    # legitimate return so it binds on the analytic-penalty blow-up channel
    # (unbounded joint-limit term in imagination).
    value_clip_min: float = -10.0
    value_clip_max: float = 1.0
    adaptive_replay: bool = False
    wm_uniform_mix: float = 0.2
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_backend: str = "tensorboard"
    seed: int = 0
    amp: str = "bf16"
    compile: bool = True
    test_hidden_dims: tuple[int, ...] | None = None


def _empty_stats() -> dict[str, list[torch.Tensor]]:
    return {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class MimicWmLearner:
    def __init__(
        self,
        env: Any,
        cfg: MimicWmConfig,
        log_dir: str | Path,
        device: str | torch.device | None = None,
    ):
        """``env`` is an rsl-rl style wrapper: ``step(action)`` returns
        ``(obs, rewards, dones, extras)`` with the mimic-wm transition under
        ``extras["mimic_wm"]``, and the pre-step current state under
        ``env.unwrapped.extras["mimic_wm_state"]``.
        """
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        if cfg.hist < 1 or cfg.horizon < 1 or cfg.pi_batch < 1:
            raise ValueError("hist, horizon and pi_batch must be positive")
        if cfg.buffer_steps < max(cfg.hist, cfg.wm_window):
            raise ValueError("buffer_steps must cover observation history and wm_window")
        if cfg.clip_value_target and not cfg.value_clip_min < cfg.value_clip_max:
            raise ValueError("value_clip_min must be below value_clip_max")
        if cfg.test_hidden_dims is not None and not cfg.test_hidden_dims:
            raise ValueError("test_hidden_dims must be nonempty when provided")

        self.env = env
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_device = torch.device(env.device)
        self.device = self.env_device if device is None else torch.device(device)
        # Host-bridge setups keep env tensors on CPU while the learner may run
        # on an accelerator; when the two differ, the learner works on its own
        # MDP copy and bridges tensors at the env boundary.
        self._bridged = self.device != self.env_device
        mdp = get_mimic_wm_mdp(env.unwrapped)
        self.mdp = mdp.to(self.device) if self._bridged else mdp
        observations = getattr(env.unwrapped, "obs_buf", {})
        for group, dim in (("policy", self.mdp.obs_dim), ("critic", self.mdp.critic_dim)):
            if group in observations and observations[group].shape[-1] != dim * cfg.hist:
                raise ValueError(f"{group} observation width does not match the learner's hist={cfg.hist}")
        self._sampler = getattr(env.unwrapped, "mimic_wm_sampler", None)
        if cfg.clip_value_target:
            # Rescale the normalized clip bounds into raw reward units.
            scale = self.mdp.max_step_reward / (1 - cfg.gamma)
            cfg.value_clip_min *= scale
            cfg.value_clip_max *= scale

        # TF32 only pays off for fp32 (amp="none") training: under bf16 autocast
        # the heavy GEMMs already run in bf16.
        torch.backends.cuda.matmul.allow_tf32 = cfg.amp == "none"
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None

        self.wm = self._make_world_model().to(self.device)
        self.pi = self._make_policy().to(self.device)
        self.critic = Critic(self._critic_input_dim(), hidden=cfg.critic_hidden).to(self.device)
        if cfg.test_hidden_dims is not None:
            self.critic.net = mlp(self._critic_input_dim(), cfg.test_hidden_dims, 1).to(
                self.device
            )
        fused_adam = self.device.type == "cuda"
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr, fused=fused_adam)
        self.opt_pi = torch.optim.Adam(
            self.pi.parameters(), lr=cfg.lr, betas=(0.9, 0.95), fused=fused_adam
        )
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, fused=fused_adam)
        self._v_mean = torch.zeros((), device=self.device)
        self._v_std = torch.ones((), device=self.device)
        self._v_ema = 0.01
        self._v_init = False

        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            env.num_envs,
            self.device,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
        )
        self.err_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((self.mdp.num_joints,), 3e-3),
                torch.full((self.mdp.num_joints,), 5e-2),
            )
        ).to(self.device)

        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        self.stats = _empty_stats()
        self._episode_returns = deque(maxlen=100)
        self._episode_lengths = deque(maxlen=100)
        self._episode_scores = deque(maxlen=100)
        self._wm_rollout_fn = self._compile(self._wm_rollout)
        self._wm_err_fn = self._compile(self._wm_err)
        self._imag_step_fn = self._compile(self._imag_step)

        with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(cfg), stream, indent=2)

    def _compile(self, function):
        if not self.cfg.compile:
            return function
        if self.device.type == "mps":
            # Inductor Metal codegen is not usable on MPS; keep AOT graph
            # capture without kernel generation.
            return torch.compile(function, backend="aot_eager")
        return torch.compile(function)

    def _make_world_model(self) -> WorldModel:
        kwargs = {
            "num_joints": self.mdp.num_joints,
            "context_dim": self.mdp.context_dim,
        }
        if self.cfg.test_hidden_dims is not None:
            return WorldModel(hidden=self.cfg.test_hidden_dims, **kwargs)
        return WorldModel(hidden=self.cfg.wm_hidden, **kwargs)

    def _make_policy(self) -> Policy:
        return Policy(
            self.mdp.obs_dim * self.cfg.hist, self.mdp.num_joints,
            self.cfg.test_hidden_dims or self.cfg.policy_hidden,
        )

    def _state_info(self) -> dict[str, torch.Tensor]:
        info: dict[str, torch.Tensor] = self.env.unwrapped.extras[MIMIC_WM_STATE_KEY]
        if self._bridged:
            info = {key: value.to(self.device) for key, value in info.items()}
        return info

    def _critic_input_dim(self) -> int:
        return self.mdp.critic_dim * self.cfg.hist

    def _wm_rollout(
        self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor
    ) -> State:
        state = unflatten_state(states[:, 0])
        predictions = []
        for index in range(self.cfg.wm_window):
            state = self.wm.step(
                state,
                actions[:, index],
                self.mdp.normalize_context(contexts[:, index]),
            )
            predictions.append(state)
        return (
            torch.stack([prediction[0] for prediction in predictions], 1),
            torch.stack([prediction[1] for prediction in predictions], 1),
            torch.stack([prediction[2] for prediction in predictions], 1),
            torch.stack([prediction[3] for prediction in predictions], 1),
            torch.stack([prediction[4] for prediction in predictions], 1),
            torch.stack([prediction[5] for prediction in predictions], 1),
        )

    def _wm_err(
        self, predicted: State, next_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        error = (self.wm.state_error(predicted, unflatten_state(next_states)) / std) ** 2
        error = error.mean(-1)
        return (error * mask).sum() / next_states.shape[0], error[:, -1]

    def _imag_step(self, state, obs_hist, frames, context):
        action = self.pi(self.mdp.pack_history(obs_hist, self.mdp.actor_dims))
        next_state = self.wm.step(state, action, self.mdp.normalize_context(context))
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, context, next_frames))
        history = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        return next_state, action, history, next_frames

    def _clip_values(self, values: torch.Tensor) -> torch.Tensor:
        """Bound value bootstraps / regression targets; NaN passes through so divergence stays visible."""
        if not self.cfg.clip_value_target:
            return values
        return values.clamp(min=self.cfg.value_clip_min, max=self.cfg.value_clip_max)

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            info = self._state_info()
            # Use the actual mimic manager output, including its noise and history.
            obs = self.env.get_observations()["policy"].to(self.device)
            action = self.pi(obs)
            # Execution noise is added inside the env action term (only the
            # setpoint and transition["a"] carry it); the clean output drives
            # a_prev, the observation action slot, and deterministic evaluation.
            if self._bridged:
                _, _, _, extras = self.env.step(action.to(self.env_device))
            else:
                _, _, _, extras = self.env.step(action)
            transition = extras[MIMIC_WM_TRANSITION_KEY]
            if self._bridged:
                transition = {key: value.to(self.device) for key, value in transition.items()}
            self.buffer.add(transition)
            initial = unflatten_state(transition["s"])
            next_state = unflatten_state(transition["s1"])
            self.wm.in_norm.update(
                self.wm.inputs(
                    initial,
                    transition["a"],
                    self.mdp.normalize_context(transition["xi"]),
                )
            )
            self.wm.out_norm.update(self.wm.targets(initial, next_state))
            self.sim_steps += self.env.num_envs
            self.stats["r"].append(transition["r"].mean() / self.mdp.max_step_reward)
            if "ep_len" in transition:
                self._episode_returns.extend(transition["ep_ret"].detach().cpu().tolist())
                self._episode_lengths.extend(transition["ep_len"].detach().cpu().tolist())
                self._episode_scores.extend(transition["ep_ret_norm"].detach().cpu().tolist())
                self.stats["ep_len"].append(transition["ep_len"].mean())
                self.stats["ep_ret"].append(transition["ep_ret_norm"].mean())
                self.stats["term"].append(transition["term"].float().sum())
                self.stats["done"].append(transition["done"].float().sum())

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.buffer.sample_windows(
            self.cfg.wm_batch,
            self.cfg.wm_window,
            weights=self._replay_weights(self.cfg.wm_uniform_mix),
        )
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        loss, last_error = self._wm_err_fn(predicted, window["s1"], window["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        mask = window["mask"][:, -1]
        mean_last_error = (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)
        return loss.detach() / self.cfg.wm_window, mean_last_error

    def _sample_valid_starts(
        self, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        starts = []
        histories = []
        count = 0
        attempts = 0
        weights = self._replay_weights(0.0)
        while count < batch_size:
            attempts += 1
            if attempts > 100:
                raise ValueError('Replay has too few reference starts for the imagination horizon')
            start, history = self.buffer.sample_starts(
                2 * batch_size, self.cfg.hist, weights=weights
            )
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon) & history["complete"]
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)

        def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)

    def _replay_weights(self, uniform_mix: float) -> torch.Tensor | None:
        """Exact segment-probability weights over replay rows, optionally uniform-flattened.

        Rows are weighted by ``p[segment] / count[segment]`` so the induced
        segment distribution matches the sampler's ``p`` rather than ``p`` times
        visitation. ``uniform_mix`` blends in row-uniform mass: the policy keeps
        its mixture pure while the world model retains broad coverage.
        """
        sampler = self._sampler
        if sampler is None or not self.cfg.adaptive_replay or self.buffer.size == 0:
            return None
        with torch.no_grad():
            probabilities = sampler.replay_segment_probabilities()
            frames = self.buffer.data["f"][: self.buffer.size]
            segments = sampler.replay_segments(frames).reshape(-1)
            counts = torch.bincount(segments, minlength=sampler.replay_num_segments()).clamp(min=1)
            weights = probabilities[segments] / counts[segments]
            weights = (1.0 - uniform_mix) * weights / weights.sum() + uniform_mix / weights.numel()
            return weights.to(self.device).reshape(self.buffer.size, self.buffer.N)

    def policy_update(self):
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        state = unflatten_state(start["s"])
        context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
        with torch.no_grad():
            obs_hist = self.mdp.add_noise(self.mdp.obs(
                unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"]
            ))
            critic_hist = self.mdp.critic_features(
                unflatten_state(history["s"]), history["f"], history["xi"], history["a_prev"]
            )
            if self.mdp.zero_history:
                obs_hist = obs_hist * history["valid"][..., None]
                critic_hist = critic_hist * history["valid"][..., None]
            else:
                # Classic history backfills with copies of the first noisy frame.
                first = history["valid"].long().argmax(-1)
                first_obs = obs_hist[torch.arange(len(first), device=self.device), first]
                obs_hist = torch.where(history["valid"][..., None], obs_hist, first_obs[:, None, :])

        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)
        critic_inputs, rewards, dones = [], [], []
        objective = torch.zeros(self.cfg.pi_batch, device=self.device)
        survival = torch.ones_like(objective)
        reward_sum = torch.zeros_like(objective)
        try:
            for index in range(self.cfg.horizon):
                critic_inputs.append(self.mdp.pack_history(critic_hist.detach(), self.mdp.critic_dims))
                old_state = state
                state, action, obs_hist, frames = self._imag_step_fn(state, obs_hist, frames, context)
                reward = self.mdp.reward(state, frames, action, previous_action)
                done = self.mdp.soft_termination(state, frames, old_state)
                objective = objective + (self.cfg.gamma**index) * survival * reward
                reward_sum = reward_sum + survival.detach() * reward.detach()
                rewards.append(reward.detach())
                dones.append(done.detach())
                survival = survival * (1 - done)
                previous_action = action
                critic_obs = self.mdp.critic_features(state, frames, context, action)
                critic_hist = torch.cat((critic_hist[:, 1:], critic_obs.unsqueeze(1)), 1)

            final_input = self.mdp.pack_history(critic_hist, self.mdp.critic_dims)
            critic_inputs.append(final_input.detach())
            if not self.cfg.no_critic:
                bootstrap = self.critic(final_input) * self._v_std + self._v_mean
                objective = objective + (self.cfg.gamma**self.cfg.horizon) * survival * self._clip_values(bootstrap)
            loss = -objective.mean()
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if torch.isfinite(grad_norm):
                self.opt_pi.step()
        finally:
            _set_requires_grad(self.wm, True)
            _set_requires_grad(self.critic, True)

        value_loss = torch.zeros((), device=self.device)
        if not self.cfg.no_critic:
            # Evaluate targets one rollout step at a time to bound peak memory.
            with torch.no_grad():
                values = [self.critic(x) * self._v_std + self._v_mean for x in critic_inputs]
                target = values[-1]
                targets = []
                for index in range(self.cfg.horizon - 1, -1, -1):
                    target = rewards[index] + self.cfg.gamma * (1 - dones[index]) * (
                        (1 - self.cfg.lam) * values[index + 1] + self.cfg.lam * target
                    )
                    targets.append(target)
                targets = self._clip_values(torch.stack(targets[::-1]))
                mean, std = targets.mean(), targets.std().clamp(min=1e-2)
                if not self._v_init:
                    self._v_mean, self._v_std = mean, std
                    self._v_init = True
                else:
                    self._v_mean = (1 - self._v_ema) * self._v_mean + self._v_ema * mean
                    self._v_std = (
                        (1 - self._v_ema) * self._v_std**2 + self._v_ema * std**2
                    ).sqrt().clamp(min=1e-2)
                targets = ((targets - self._v_mean) / self._v_std).clamp(-20, 20).flatten()

            flat_input = torch.cat(critic_inputs[:-1], 0)
            if self.cfg.critic_minibatches > len(flat_input):
                raise ValueError("critic_minibatches exceeds imagined transitions")
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(len(flat_input), device=self.device)
                for indices in permutation.tensor_split(self.cfg.critic_minibatches):
                    value_loss = ((self.critic(flat_input[indices]) - targets[indices]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()

        imagined_reward = (reward_sum / self.cfg.horizon / self.mdp.max_step_reward).mean()
        return loss.detach(), value_loss.detach(), imagined_reward, grad_norm.detach(), survival.mean().detach()

    def lr_at(self, iteration: int) -> float:
        if iteration >= self.cfg.lr_decay_iters:
            return self.cfg.lr * 0.03
        return self.cfg.lr * (
            0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / self.cfg.lr_decay_iters))
        )

    def _checkpoint(self, include_optimizers: bool) -> dict[str, Any]:
        checkpoint = {
            "format_version": 2,
            "v_mean": self._v_mean,
            "v_std": self._v_std,
            "v_init": self._v_init,
            "wm": self.wm.state_dict(),
            "pi": self.pi.state_dict(),
            "iteration": self.iteration,
            "wall_time": self.wall_time,
            "sim_steps": self.sim_steps,
            "args": asdict(self.cfg),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self._sampler is not None:
            checkpoint["sampler"] = {
                name: tensor.detach().cpu()
                for name, tensor in self._sampler.sampler_state().items()
            }
        if include_optimizers:
            checkpoint.update(
                {
                    "critic": self.critic.state_dict(),
                    "opt_wm": self.opt_wm.state_dict(),
                    "opt_pi": self.opt_pi.state_dict(),
                    "opt_v": self.opt_v.state_dict(),
                }
            )
        return checkpoint

    def save(self, path: str | Path | None = None, include_optimizers: bool = True) -> Path:
        destination = Path(path) if path is not None else self.log_dir / "last.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint(include_optimizers), temporary)
        os.replace(temporary, destination)
        return destination

    def load(self, path: str | Path, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get('format_version') != 2:
            raise ValueError('This checkpoint uses the old goal-conditioned task; start a new mimic-wm run')
        self._v_mean=checkpoint['v_mean']
        self._v_std=checkpoint['v_std']
        self._v_init=checkpoint['v_init']
        required = {"critic", "opt_wm", "opt_pi", "opt_v"} if load_optimizers else set()
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(
                f"mimic-wm resume requires a full last.pt checkpoint; missing {missing} in {path}"
            )
        self.wm.load_state_dict(checkpoint["wm"])
        self.pi.load_state_dict(checkpoint["pi"])
        if 'critic' in checkpoint: self.critic.load_state_dict(checkpoint['critic'])
        if load_optimizers:
            self.opt_wm.load_state_dict(checkpoint['opt_wm'])
            self.opt_pi.load_state_dict(checkpoint['opt_pi'])
            self.opt_v.load_state_dict(checkpoint['opt_v'])
        sampler_state = checkpoint.get("sampler")
        if sampler_state is not None:
            if self._sampler is None:
                raise ValueError(
                    f"checkpoint carries sampler state but {type(self.env).__name__} exposes none"
                )
            self._sampler.load_sampler_state(sampler_state)
        self.iteration = checkpoint["iteration"]
        self.wall_time = checkpoint["wall_time"]
        self.sim_steps = checkpoint["sim_steps"]

    @staticmethod
    def _mean(values: list[torch.Tensor]) -> float:
        return float(sum(float(value) for value in values) / max(len(values), 1))

    def train(self, resume: str | Path | None = None) -> dict[str, float | int | bool]:
        if resume is not None:
            self.load(resume)
        start_iteration = self.iteration
        self.stats = _empty_stats()
        print(f"[mimic-wm]: training for {self.cfg.iterations} iterations")
        if start_iteration == 0:
            print(f"[mimic-wm]: warming up with {self.cfg.warmup_steps} collect steps")
            self.collect(self.cfg.warmup_steps)
        required_steps = max(self.cfg.hist, self.cfg.wm_window)
        if start_iteration > 0:
            required_steps = max(required_steps, self.cfg.warmup_steps)
        if self.buffer.size < required_steps:
            self.collect(required_steps - self.buffer.size)
        print(f"[mimic-wm]: replay buffer filled to {self.buffer.size}/{self.cfg.buffer_steps}")

        summary_writer = None
        if self.cfg.log_backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(str(self.log_dir), max_queue=1, flush_secs=60)
            except ImportError:
                print("[mimic-wm]: tensorboard unavailable; logging to console only")

        try:
            train_start = time.time() - self.wall_time
            step_accumulator = 0.0
            recent_iteration_times = deque(maxlen=100)
            last_phase = None
            last_metrics: dict[str, float | int | None] = {}
            for iteration in range(start_iteration, self.cfg.iterations):
                if self.cfg.compile and iteration == start_iteration:
                    print("[mimic-wm]: compiling world model (one-off, slow on first run)")
                if self.cfg.compile and iteration == self.cfg.wm_only_iters:
                    print("[mimic-wm]: compiling policy update (one-off, slow on first run)")
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize(self.device)
                iteration_start = time.time()
                learning_rate = self.lr_at(iteration)
                for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                step_accumulator += 1.0 / self.cfg.utd
                collect_steps = int(step_accumulator)
                step_accumulator -= collect_steps
                stats_mark = len(self.stats["r"])
                self.collect(collect_steps)
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize(self.device)
                train_phase_start = time.time()
                collect_time = train_phase_start - iteration_start
                wm_loss = torch.zeros((), device=self.device)
                wm_error = torch.zeros((), device=self.device)
                for _ in range(self.cfg.wm_batches):
                    wm_loss, wm_error = self.wm_update()
                pi_loss: float | None = None
                v_loss: float | None = None
                imag_rew: float | None = None
                pi_gradnorm: float | None = None
                imag_survival: float | None = None
                if iteration >= self.cfg.wm_only_iters:
                    policy_loss, value_loss, imagined_reward, grad_norm, survival = (
                        self.policy_update()
                    )
                    pi_loss = float(policy_loss)
                    v_loss = float(value_loss)
                    imag_rew = float(imagined_reward)
                    pi_gradnorm = float(grad_norm)
                    imag_survival = float(survival)
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize(self.device)
                train_time = time.time() - train_phase_start
                self.iteration = iteration + 1
                self.wall_time = time.time() - train_start

                iter_rewards = self.stats["r"][stats_mark:]
                mean_reward = self._mean(iter_rewards) if iter_rewards else None
                mean_episode_reward = (
                    sum(self._episode_returns) / len(self._episode_returns) if self._episode_returns else None
                )
                mean_episode_length = (
                    sum(self._episode_lengths) / len(self._episode_lengths) if self._episode_lengths else None
                )
                mean_episode_score = (
                    sum(self._episode_scores) / len(self._episode_scores) if self._episode_scores else None
                )
                iteration_time = time.time() - iteration_start
                fps = collect_steps * self.env.num_envs / max(iteration_time, 1e-9)
                phase = "world model warmup" if pi_loss is None else "world model + policy"
                if phase != last_phase:
                    recent_iteration_times.clear()
                    last_phase = phase
                recent_iteration_times.append(iteration_time)
                mean_iteration_time = sum(recent_iteration_times) / len(recent_iteration_times)
                remaining = (self.cfg.iterations - self.iteration) * mean_iteration_time
                eta_note = " (warmup rate)" if pi_loss is None else ""
                lines = [
                    f"{' Learning iteration ' + str(self.iteration) + '/' + str(self.cfg.iterations) + ' ':#^80}",
                    f"{'Phase:':>28} {phase}",
                    f"{'Computation:':>28} {fps:.0f} steps/s ({iteration_time:.2f}s/iter)",
                    f"{'Collection time:':>28} {collect_time:.3f}s",
                    f"{'Learning time:':>28} {train_time:.3f}s",
                    f"{'World model loss:':>28} {float(wm_loss):.6f}",
                    f"{'World model rollout error:':>28} {float(wm_error):.6f}",
                ]
                if pi_loss is not None:
                    lines.extend([
                        f"{'Policy loss:':>28} {pi_loss:.6f}",
                        f"{'Value loss:':>28} {v_loss:.6f}",
                        f"{'Imagined reward (norm):':>28} {imag_rew:.4f}",
                        f"{'Imagined survival:':>28} {imag_survival:.4f}",
                    ])
                if mean_reward is not None:
                    lines.append(f"{'Simulation reward (norm):':>28} {mean_reward:.4f}")
                if mean_episode_reward is not None:
                    lines.append(f"{'Mean reward:':>28} {mean_episode_reward:.4f}")
                if mean_episode_length is not None:
                    lines.append(f"{'Mean episode length:':>28} {mean_episode_length:.2f}")
                lines.append(f"{'Total simulation steps:':>28} {self.sim_steps}")
                lines.append(f"{'Total training time:':>28} {timedelta(seconds=int(self.wall_time))}")
                lines.append(f"{'ETA:':>28} {timedelta(seconds=int(remaining))}{eta_note}")
                print("\n".join(lines), flush=True)
                if summary_writer is not None:
                    summary = {
                        "train/wm_loss": float(wm_loss),
                        "train/wm_err8": float(wm_error),
                        "train/lr": learning_rate,
                        "train/sim_rew": mean_reward,
                        "Train/mean_reward": mean_episode_reward,
                        "Train/mean_episode_length": mean_episode_length,
                        "train/ep_ret_norm": mean_episode_score,
                        "train/collect_steps": collect_steps,
                        "train/collect_time": collect_time,
                        "train/learn_time": train_time,
                        "train/iteration_time": iteration_time,
                    }
                    if pi_loss is not None:
                        summary.update(
                            {
                                "train/pi_loss": pi_loss,
                                "train/v_loss": v_loss,
                                "train/imag_rew": imag_rew,
                                "train/pi_gradnorm": pi_gradnorm,
                                "train/imag_survival": imag_survival,
                            }
                        )
                    for tag, value in summary.items():
                        if value is not None:
                            summary_writer.add_scalar(tag, value, self.iteration)

                last_metrics = {
                    "wm_loss": float(wm_loss),
                    "wm_err8": float(wm_error),
                    "sim_rew": mean_reward,
                    "pi_loss": pi_loss,
                    "v_loss": v_loss,
                }
                self.stats = _empty_stats()

                if (
                    self.iteration % self.cfg.ckpt_every == 0
                    or self.iteration == self.cfg.iterations
                ):
                    self.wall_time = time.time() - train_start
                    self.save()
                    numbered = self.save(
                        self.log_dir / f"ckpt_{self.iteration:06d}.pt",
                        include_optimizers=False,
                    )
                    print(f"[mimic-wm]: saved checkpoint {numbered}")

            self.wall_time = time.time() - train_start
            return {
                "completed": self.iteration >= self.cfg.iterations,
                "iteration": self.iteration,
                "sim_steps": self.sim_steps,
                "wall_time": self.wall_time,
                **last_metrics,
            }
        finally:
            if summary_writer is not None:
                summary_writer.close()


__all__ = ["MimicWmConfig", "MimicWmLearner"]
