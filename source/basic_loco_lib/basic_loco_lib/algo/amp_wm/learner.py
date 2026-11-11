"""Train a locomotion policy through learned dynamics with AMP's style reward.

The collect / WM-fit / imagination schedule follows the same model-based
recipe as mimic-wm. The policy is
optimised purely in imagination; the reward mixes the AMP discriminator's
style reward with differentiable velocity-tracking terms. The discriminator is
trained on *real* transitions from the world-model replay buffer against the
expert motion dataset, so it never sees imagined data.

This module is self-contained: it does not subclass or import any other task's
learner, so AMP-WM can evolve independently.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from . import networks
from .contracts import (
    AMP_WM_STATE_KEY,
    AMP_WM_TRANSITION_KEY,
    State,
    get_amp_wm_mdp,
)
from .discriminator import Discriminator
from .geometry import unflatten_state
from .motion_loader import AMPLoader
from .networks import Critic, Policy, WorldModel, mlp, run_mlp
from .normalizer import Normalizer
from .replay import ReplayBuffer


def _run_with_compile_bar(label: str, estimate_s: int, fn, *args, **kwargs):
    """Run ``fn`` once while rendering a time-estimated compile progress bar.

    ``torch.compile`` blocks on the first call, so a daemon thread draws the
    bar while the main thread compiles. The bar is only an estimate; it jumps
    to 100% when the call returns.
    """
    width = 30
    start = time.perf_counter()
    done = threading.Event()

    def render() -> None:
        while not done.wait(1.0):
            elapsed = time.perf_counter() - start
            frac = min(elapsed / estimate_s, 0.99)
            filled = int(round(frac * width))
            eta = max(estimate_s - int(elapsed), 0)
            sys.stdout.write(
                f"\r{label} |{'█' * filled}{'░' * (width - filled)}| "
                f"{int(round(frac * 100)):3d}% {int(elapsed):5d}s  ETA {eta}s  "
            )
            sys.stdout.flush()

    thread = threading.Thread(target=render, daemon=True)
    thread.start()
    try:
        return fn(*args, **kwargs)
    finally:
        done.set()
        thread.join()
        elapsed = time.perf_counter() - start
        sys.stdout.write(
            f"\r{label} |{'█' * width}| 100% {int(elapsed):5d}s  "
            f"done, total {int(elapsed)}s  \n"
        )
        sys.stdout.flush()


@dataclass
class AmpWmConfig:
    iterations: int = 100_001
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
    wm_contact_weight: float = 0.1
    wm_statistics: str = "fixed"
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    clip_value_target: bool = True
    # Value-clip bounds expressed as multiples of the task's value-function
    # supremum ``max_step_reward / (1 - gamma)``; rescaled in ``__init__``.
    value_clip_min: float = -10.0
    value_clip_max: float = 1.0
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_backend: str = "tensorboard"
    seed: int = 0
    # ``amp`` is the mixed-precision mode ("none" or "bf16"), not the AMP
    # discriminator coefficient.
    amp: str = "none"
    compile: bool = True
    test_hidden_dims: tuple[int, ...] | None = None

    # AMP discriminator and expert data.
    amp_reward_coef: float = 0.1
    amp_motion_files: str = ""
    amp_task_reward_lerp: float = 0.75
    amp_discr_hidden_dims: tuple[int, ...] = (1024, 512, 256)
    amp_body_names: tuple[str, ...] = ()
    amp_anchor_name: str = ""
    amp_batch: int = 4096
    amp_updates: int = 1
    amp_lr: float = 1.0e-3
    amp_weight_decay_trunk: float = 1.0e-3
    amp_weight_decay_out: float = 1.0e-2
    amp_grad_pen_lambda: float = 10.0


def _empty_stats() -> dict[str, list[torch.Tensor]]:
    return {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class AmpWmLearner:
    def __init__(
        self,
        env: Any,
        cfg: AmpWmConfig,
        log_dir: str | Path,
        device: str | torch.device | None = None,
        *,
        amp_enabled: bool = True,
    ):
        """``env`` is an rsl-rl style wrapper: ``step(action)`` returns
        ``(obs, rewards, dones, extras)`` with the AMP-WM transition under
        ``extras["amp_wm"]`` and the pre-step state under
        ``env.unwrapped.extras["amp_wm_state"]``.
        """
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.wm_statistics not in ("fixed", "running"):
            raise ValueError(f"wm_statistics must be 'fixed' or 'running', got {cfg.wm_statistics!r}")
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
        mdp = get_amp_wm_mdp(env.unwrapped)
        self.mdp = mdp.to(self.device) if self._bridged else mdp
        observations = getattr(env.unwrapped, "obs_buf", {})
        for group, dim in (("policy", self.mdp.obs_dim), ("critic", self.mdp.critic_dim)):
            if group in observations and observations[group].shape[-1] != dim * cfg.hist:
                raise ValueError(f"{group} observation width does not match the learner's hist={cfg.hist}")
        if cfg.clip_value_target:
            # Rescale the normalized clip bounds into raw reward units.
            scale = self.mdp.max_step_reward / (1 - cfg.gamma)
            cfg.value_clip_min *= scale
            cfg.value_clip_max *= scale

        torch.backends.cuda.matmul.allow_tf32 = False
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
        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        self.stats = _empty_stats()
        self._episode_returns = deque(maxlen=100)
        self._episode_lengths = deque(maxlen=100)
        self._episode_scores = deque(maxlen=100)
        self._episode_metrics: dict[str, list[torch.Tensor]] = {}
        self._wm_rollout_fn = self._compile(self._wm_rollout)
        self._wm_err_fn = self._compile(self._wm_err)
        self._imag_step_fn = self._compile(self._imag_step)
        self._reward_term_fn = self._compile(self._reward_term)
        self._critic_feat_fn = self._compile(self._critic_feat)
        self._recovery_step_fn = self._compile(self.mdp.recovery_step)
        # First-call flags drive the one-off compile progress bars.
        self._wm_update_compiled = not self.cfg.compile
        self._pi_update_compiled = not self.cfg.compile

        self._amp_enabled = amp_enabled
        self._amp_metrics: dict[str, float] = {}
        self.discriminator = None
        self.amp_normalizer = None
        self.amp_data = None
        self.opt_amp = None
        if amp_enabled:
            self._init_amp()

        with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(cfg), stream, indent=2)

    def _init_amp(self) -> None:
        cfg = self.cfg
        if not cfg.amp_motion_files:
            raise ValueError("amp-wm requires amp_motion_files")
        body_names = tuple(self.mdp.amp_body_names)
        if tuple(cfg.amp_body_names) != body_names:
            raise ValueError(
                f"amp-wm body order mismatch: cfg={tuple(cfg.amp_body_names)} mdp={body_names}"
            )
        if cfg.amp_anchor_name != self.mdp.amp_anchor_name:
            raise ValueError(
                f"amp-wm anchor mismatch: cfg={cfg.amp_anchor_name!r} "
                f"mdp={self.mdp.amp_anchor_name!r}"
            )

        self.amp_data = AMPLoader(
            motion_file=cfg.amp_motion_files,
            body_names=list(body_names),
            anchor_name=cfg.amp_anchor_name,
            device=str(self.device),
        )
        if self.amp_data.observation_dim != self.mdp.amp_feature_dim:
            raise ValueError(
                "AMP feature width mismatch: expert "
                f"{self.amp_data.observation_dim} vs mdp {self.mdp.amp_feature_dim}"
            )
        self.amp_normalizer = Normalizer(self.amp_data.observation_dim, device=self.device)
        self.discriminator = Discriminator(
            2 * self.amp_data.observation_dim,
            cfg.amp_reward_coef,
            list(cfg.amp_discr_hidden_dims),
            self.device,
            cfg.amp_task_reward_lerp,
        ).to(self.device)
        self.opt_amp = torch.optim.Adam(
            [
                {
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": cfg.amp_weight_decay_trunk,
                },
                {
                    "params": self.discriminator.amp_linear.parameters(),
                    "weight_decay": cfg.amp_weight_decay_out,
                },
            ],
            lr=cfg.amp_lr,
            betas=(0.9, 0.999),
            fused=self.device.type == "cuda",
        )
        self._amp_batch = int(cfg.amp_batch)
        self._amp_updates = int(cfg.amp_updates)

    def _compile(self, function):
        if not self.cfg.compile:
            return function
        if self.device.type == "mps":
            return torch.compile(function, backend="aot_eager")
        return torch.compile(function)

    def _make_world_model(self) -> WorldModel:
        kwargs = {
            "num_joints": self.mdp.num_joints,
            "context_dim": self.mdp.wm_context_dim,
            "statistics_mode": self.cfg.wm_statistics,
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
        info: dict[str, torch.Tensor] = self.env.unwrapped.extras[AMP_WM_STATE_KEY]
        if self._bridged:
            info = {key: value.to(self.device) for key, value in info.items()}
        return info

    def _critic_input_dim(self) -> int:
        return self.mdp.critic_dim * self.cfg.hist

    def _pack_critic(self, history):
        return self.mdp.pack_history(history, self.mdp.critic_dims)

    def _wm_rollout(
        self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor
    ) -> tuple[State, torch.Tensor]:
        state = unflatten_state(states[:, 0])
        predictions = []
        contacts = []
        for index in range(self.cfg.wm_window):
            state, contact = self.wm.step(
                state,
                actions[:, index],
                self.mdp.normalize_context(contexts[:, index]),
            )
            predictions.append(state)
            contacts.append(contact)
        return (
            (
                torch.stack([prediction[0] for prediction in predictions], 1),
                torch.stack([prediction[1] for prediction in predictions], 1),
                torch.stack([prediction[2] for prediction in predictions], 1),
                torch.stack([prediction[3] for prediction in predictions], 1),
                torch.stack([prediction[4] for prediction in predictions], 1),
                torch.stack([prediction[5] for prediction in predictions], 1),
            ),
            torch.stack(contacts, 1),
        )

    def _wm_err(
        self,
        predicted: State,
        contacts: torch.Tensor,
        next_states: torch.Tensor,
        contact_targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = self.wm.error_scale()
        error = (self.wm.state_error(predicted, unflatten_state(next_states)) / std) ** 2
        error = error.mean(-1)
        state_loss = (error * mask).sum() / next_states.shape[0]
        contact_loss = nn.functional.binary_cross_entropy_with_logits(
            contacts, contact_targets, reduction="none"
        ).mean(-1)
        contact_loss = (contact_loss * mask).sum() / next_states.shape[0]
        return state_loss + self.cfg.wm_contact_weight * contact_loss, error[:, -1]

    def _imag_step(self, state, obs_hist, frames, context):
        action = self.pi(self.mdp.pack_history(obs_hist, self.mdp.actor_dims))
        next_state, contact = self.wm.step(state, action, self.mdp.normalize_context(context))
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, context, next_frames))
        history = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        return next_state, action, history, next_frames, contact

    def _reward_term(
        self,
        state: State,
        prev_state: State,
        frames: torch.Tensor,
        actions: torch.Tensor,
        previous_actions: torch.Tensor,
        context: torch.Tensor,
        recovery_gate: torch.Tensor,
        contacts: torch.Tensor,
    ) -> torch.Tensor:
        """Task/style rewards for a flattened horizon."""
        reward = self.mdp.reward(
            state, frames, actions, previous_actions,
            context=context, prev_state=prev_state,
            recovery_gate=recovery_gate, contacts=contacts,
        )
        if self._amp_enabled:
            style_reward = self._amp_style_reward(prev_state, state)
            lerp = self.cfg.amp_task_reward_lerp
            reward = (1.0 - lerp) * style_reward + lerp * reward
        return reward

    def _critic_feat(
        self,
        state: State,
        frames: torch.Tensor,
        context: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Critic features for a horizon-flattened batch."""
        return self.mdp.critic_features(state, frames, context, previous_actions)

    def _clip_values(self, values: torch.Tensor) -> torch.Tensor:
        """Bound value bootstraps / targets; NaN passes through so divergence stays visible."""
        if not self.cfg.clip_value_target:
            return values
        return values.clamp(min=self.cfg.value_clip_min, max=self.cfg.value_clip_max)

    # ------------------------------------------------------------------
    # AMP discriminator
    # ------------------------------------------------------------------

    def _amp_style_reward(self, old_state, state) -> torch.Tensor:
        """Discriminator style reward for an imagined ``state -> next`` pair."""
        normalized = self.amp_normalizer
        feature_old = normalized.normalize_torch(self.mdp.amp_features(old_state), self.device)
        feature_new = normalized.normalize_torch(self.mdp.amp_features(state), self.device)
        score = run_mlp(self.discriminator, torch.cat((feature_old, feature_new), -1))
        style = torch.clamp(1.0 - 0.25 * (score - 1.0) ** 2, min=0.0)
        return self.cfg.amp_reward_coef * style.squeeze(-1)

    def amp_update(self) -> dict[str, float]:
        """One discriminator update on real replay transitions."""
        batch = self._amp_batch
        window = self.buffer.sample_windows(batch, 1)
        states = unflatten_state(window["s"][:, 0])
        next_states = unflatten_state(window["s1"][:, 0])
        with torch.no_grad():
            policy_state = self.mdp.amp_features(states)
            policy_next_state = self.mdp.amp_features(next_states)
        expert_state, expert_next_state = next(
            self.amp_data.feed_forward_generator(1, batch)
        )
        raw_expert_state, raw_expert_next_state = expert_state, expert_next_state

        with torch.no_grad():
            self.amp_normalizer.update(
                torch.cat(
                    (policy_state, policy_next_state, expert_state, expert_next_state),
                    dim=0,
                )
            )
        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

        policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
        amp_loss = 0.5 * (
            nn.MSELoss()(expert_d, torch.ones_like(expert_d))
            + nn.MSELoss()(policy_d, -torch.ones_like(policy_d))
        )
        grad_pen = self.discriminator.compute_grad_pen(
            raw_expert_state,
            raw_expert_next_state,
            lambda_=self.cfg.amp_grad_pen_lambda,
        )
        loss = amp_loss + grad_pen
        self.opt_amp.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.opt_amp.step()
        return {
            "train/amp_loss": float(amp_loss.detach()),
            "train/amp_grad_pen": float(grad_pen.detach()),
            "train/amp_policy_d": float(policy_d.detach().mean()),
            "train/amp_expert_d": float(expert_d.detach().mean()),
        }

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            info = self._state_info()
            obs = self.env.get_observations()["policy"].to(self.device)
            action = self.pi(obs)
            # Execution noise is added inside the env action term (only the
            # setpoint and transition["a"] carry it); the clean output drives
            # a_prev, the observation action slot, and deterministic evaluation.
            if self._bridged:
                _, _, _, extras = self.env.step(action.to(self.env_device))
            else:
                _, _, _, extras = self.env.step(action)
            transition = extras[AMP_WM_TRANSITION_KEY]
            if self._bridged:
                transition = {key: value.to(self.device) for key, value in transition.items()}
            # Physics-solver overflows can emit non-finite states; one such
            # transition poisons the world model and the run never recovers.
            finite = (
                torch.isfinite(transition["s"]).all(-1)
                & torch.isfinite(transition["s1"]).all(-1)
            )
            if not bool(finite.all()):
                self.stats.setdefault("nonfinite", []).append(int((~finite).sum()))
                transition = {
                    key: value[finite] if value.shape[:1] == finite.shape else value
                    for key, value in transition.items()
                }
            self.buffer.add(transition)
            self.pi.update_normalization(obs)
            initial = unflatten_state(transition["s"])
            next_state = unflatten_state(transition["s1"])
            self.wm.update_stats(
                initial,
                transition["a"],
                next_state,
                self.mdp.normalize_context(transition["xi"]),
            )
            self.sim_steps += self.env.num_envs
            self.stats["r"].append(transition["r"].mean() / self.mdp.max_step_reward)
            if "ep_len" in transition:
                # Manager logs describe completed episodes. Only capture them on
                # reset steps: env.extras may retain the previous reset's log.
                episode_log = extras.get("episode", extras.get("log", {}))
                for key, value in episode_log.items():
                    value = torch.as_tensor(value, device=self.device).detach().float().reshape(-1).clone()
                    if value.numel():
                        tag = key if "/" in key else f"Episode/{key}"
                        self._episode_metrics.setdefault(tag, []).append(value)
                self._episode_returns.extend(transition["ep_ret"].detach().cpu().tolist())
                self._episode_lengths.extend(transition["ep_len"].detach().cpu().tolist())
                self._episode_scores.extend(transition["ep_ret_norm"].detach().cpu().tolist())
                self.stats["ep_len"].append(transition["ep_len"].mean())
                self.stats["ep_ret"].append(transition["ep_ret_norm"].mean())
                self.stats["term"].append(transition["term"].float().sum())
                self.stats["done"].append(transition["done"].float().sum())

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._wm_update_compiled:
            self._wm_update_compiled = True
            return _run_with_compile_bar(
                "[amp-wm]: compiling world model", 90, self._wm_update
            )
        return self._wm_update()

    def _wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        predicted, contacts = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        loss, last_error = self._wm_err_fn(
            predicted, contacts, window["s1"], window["contact"], window["mask"]
        )
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
        while count < batch_size:
            attempts += 1
            if attempts > 100:
                raise ValueError("Replay has too few valid starts for the imagination horizon")
            start, history = self.buffer.sample_starts(2 * batch_size, self.cfg.hist)
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon) & history["complete"]
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)

        def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)

    def policy_update(self):
        if not self._pi_update_compiled:
            self._pi_update_compiled = True
            return _run_with_compile_bar(
                "[amp-wm]: compiling policy update", 180, self._policy_update
            )
        return self._policy_update()

    def _policy_update(self):
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        state = unflatten_state(start["s"])
        recovery = start["recovery"]
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
        if self._amp_enabled:
            _set_requires_grad(self.discriminator, False)
        horizon, batch_size = self.cfg.horizon, self.cfg.pi_batch
        initial_state = state
        states, actions, previous_actions, all_frames, all_contacts = [], [], [], [], []
        all_dones, recovery_gates = [], []
        try:
            # Roll out first; rewards, terminations and critic features are
            # evaluated afterwards on horizon-flattened batches (the per-step
            # eager loop they used to live in was the dominant cost).
            for _ in range(horizon):
                previous_actions.append(previous_action)
                state, action, obs_hist, frames, contact = self._imag_step_fn(
                    state, obs_hist, frames, context
                )
                recovery, done, recovery_gate = self._recovery_step_fn(state, recovery)
                all_dones.append(done)
                recovery_gates.append(recovery_gate)
                states.append(state)
                all_contacts.append(contact)
                actions.append(action)
                all_frames.append(frames)
                previous_action = action

            def flatten(parts: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(parts).flatten(0, 1)

            all_states: State = tuple(
                flatten([item[index] for item in states]) for index in range(6)
            )
            prev_states: State = tuple(
                flatten([initial_state[index]] + [item[index] for item in states[:-1]])
                for index in range(6)
            )
            flat_frames = flatten(all_frames)
            flat_actions, flat_prev_actions = flatten(actions), flatten(previous_actions)
            flat_contacts = flatten(all_contacts)

            flat_context = context.unsqueeze(0).expand(horizon, -1, -1).flatten(0, 1)
            rewards = self._reward_term_fn(
                all_states, prev_states, flat_frames, flat_actions, flat_prev_actions, flat_context,
                flatten(recovery_gates), flat_contacts,
            )
            rewards = rewards.view(horizon, batch_size)
            dones = torch.stack(all_dones)
            # AMP's is_terminated=-200: the task reward is mixed at ``lerp``,
            # so apply the same scaling here.
            rewards = rewards + self.cfg.amp_task_reward_lerp * self.mdp.fall_penalty(dones)
            alive = torch.cumprod(1 - dones, 0)
            continuation = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), 0)
            discount = self.cfg.gamma ** torch.arange(
                horizon, device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            objective = (discount * continuation * rewards).sum(0)
            survival = alive[-1]
            reward_sum = (continuation * rewards).detach().sum(0)
            detached_rewards, detached_dones = rewards.detach(), dones.detach()

            # Critic windows: concatenate the replay-history features with the
            # imagined ones, then gather all horizon+1 windows in one shot.
            # Features of s[t+1] carry a[t], the action that produced that state.
            feats = self._critic_feat_fn(
                all_states, flat_frames, flat_context, flat_actions
            ).view(horizon, batch_size, -1)
            window_rows = (
                torch.arange(horizon + 1, device=self.device).unsqueeze(1)
                + torch.arange(self.cfg.hist, device=self.device).unsqueeze(0)
            )
            windows = torch.cat((critic_hist.transpose(0, 1), feats), 0)[window_rows]
            critic_input = self._pack_critic(windows.permute(0, 2, 1, 3))
            final_input = critic_input[-1]
            if not self.cfg.no_critic:
                bootstrap = self.critic(final_input) * self._v_std + self._v_mean
                objective = objective + (self.cfg.gamma**horizon) * survival * self._clip_values(bootstrap)
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
            if self._amp_enabled:
                _set_requires_grad(self.discriminator, True)

        value_loss = torch.zeros((), device=self.device)
        if not self.cfg.no_critic:
            # One batched critic forward over all horizon+1 windows; the
            # recursion below then only chains small per-step scalars.
            with torch.no_grad():
                values = (
                    self.critic(critic_input.detach().flatten(0, 1)).view(horizon + 1, batch_size)
                    * self._v_std
                    + self._v_mean
                )
                target = values[-1]
                targets = []
                for index in range(horizon - 1, -1, -1):
                    target = detached_rewards[index] + self.cfg.gamma * (1 - detached_dones[index]) * (
                        (1 - self.cfg.lam) * values[index + 1] + self.cfg.lam * target
                    )
                    targets.append(target)
                targets = self._clip_values(torch.stack(targets[::-1]))
                valid_values = continuation.detach()
                mass = valid_values.sum().clamp(min=1)
                mean = (targets * valid_values).sum() / mass
                std = (((targets - mean).square() * valid_values).sum() / mass).sqrt().clamp(min=1e-2)
                if not self._v_init:
                    self._v_mean, self._v_std = mean, std
                    self._v_init = True
                else:
                    self._v_mean = (1 - self._v_ema) * self._v_mean + self._v_ema * mean
                    self._v_std = (
                        (1 - self._v_ema) * self._v_std**2 + self._v_ema * std**2
                    ).sqrt().clamp(min=1e-2)
                targets = ((targets - self._v_mean) / self._v_std).clamp(-20, 20).flatten()

            flat_input = critic_input[:-1].detach().reshape(-1, critic_input.shape[-1])
            critic_weights = continuation.detach().flatten()
            if self.cfg.critic_minibatches > len(flat_input):
                raise ValueError("critic_minibatches exceeds imagined transitions")
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(len(flat_input), device=self.device)
                for indices in permutation.tensor_split(self.cfg.critic_minibatches):
                    w = critic_weights[indices]
                    value_loss = (
                        ((self.critic(flat_input[indices]) - targets[indices]) ** 2) * w
                    ).sum() / w.sum().clamp(min=1)
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
            "format_version": 3,
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
        if self._amp_enabled:
            checkpoint["amp"] = {
                "discriminator": self.discriminator.state_dict(),
                "normalizer_mean": self.amp_normalizer.mean.detach().cpu(),
                "normalizer_var": self.amp_normalizer.var.detach().cpu(),
                "normalizer_count": self.amp_normalizer.count.detach().cpu(),
                "expert_cursor": int(self.amp_data._expert_cursor),
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
            if self._amp_enabled:
                checkpoint["opt_amp"] = self.opt_amp.state_dict()
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
        if checkpoint.get("format_version") not in (2, 3):
            raise ValueError("Unsupported amp-wm checkpoint format; start a new run")
        self._v_mean = checkpoint["v_mean"]
        self._v_std = checkpoint["v_std"]
        self._v_init = checkpoint["v_init"]
        required = {"critic", "opt_wm", "opt_pi", "opt_v"} if load_optimizers else set()
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(
                f"amp-wm resume requires a full last.pt checkpoint; missing {missing} in {path}"
            )
        self.wm.load_state_dict(checkpoint["wm"])
        self.pi.load_state_dict(checkpoint["pi"])
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            self.opt_wm.load_state_dict(checkpoint["opt_wm"])
            self.opt_pi.load_state_dict(checkpoint["opt_pi"])
            self.opt_v.load_state_dict(checkpoint["opt_v"])

        amp = checkpoint.get("amp")
        if self._amp_enabled and amp is not None:
            self.discriminator.load_state_dict(amp["discriminator"])
            self.amp_normalizer.mean = amp["normalizer_mean"].to(self.device)
            self.amp_normalizer.var = amp["normalizer_var"].to(self.device)
            self.amp_normalizer.count = amp["normalizer_count"].to(self.device)
            self.amp_data._expert_cursor = int(amp.get("expert_cursor", 0))
            if load_optimizers and "opt_amp" in checkpoint:
                self.opt_amp.load_state_dict(checkpoint["opt_amp"])
        elif self._amp_enabled:
            print("[amp-wm]: checkpoint has no AMP state; discriminator starts fresh")

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
        print(f"[amp-wm]: training for {self.cfg.iterations} iterations")
        if start_iteration == 0:
            print(f"[amp-wm]: warming up with {self.cfg.warmup_steps} collect steps")
            self.collect(self.cfg.warmup_steps)
        required_steps = max(self.cfg.hist, self.cfg.wm_window)
        if start_iteration > 0:
            required_steps = max(required_steps, self.cfg.warmup_steps)
        if self.buffer.size < required_steps:
            self.collect(required_steps - self.buffer.size)
        if self.wm.statistics_mode == "fixed" and not bool(self.wm.stats_fitted):
            window = self.buffer.sample_windows(
                min(8192, self.buffer.size * self.buffer.N), 1
            )
            self.wm.fit_stats(
                unflatten_state(window["s"][:, 0]),
                unflatten_state(window["s1"][:, 0]),
                window["a"][:, 0],
                self.mdp.normalize_context(window["xi"][:, 0]),
            )
        print(f"[amp-wm]: replay buffer filled to {self.buffer.size}/{self.cfg.buffer_steps}")

        summary_writer = None
        if self.cfg.log_backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(str(self.log_dir), max_queue=1, flush_secs=60)
            except ImportError:
                print("[amp-wm]: tensorboard unavailable; logging to console only")

        try:
            train_start = time.time() - self.wall_time
            step_accumulator = 0.0
            recent_iteration_times = deque(maxlen=100)
            last_phase = None
            last_metrics: dict[str, float | int | None] = {}
            for iteration in range(start_iteration, self.cfg.iterations):
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
                amp_metrics: dict[str, float] = {}
                if self._amp_enabled and self.buffer.size > 0:
                    for _ in range(self._amp_updates):
                        amp_metrics = self.amp_update()
                self._amp_metrics = amp_metrics
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
                episode_metrics = {
                    tag: torch.cat(values).mean().item()
                    for tag, values in self._episode_metrics.items()
                }
                lines = [
                    f"{' Learning iteration ' + str(self.iteration) + '/' + str(self.cfg.iterations) + ' ':#^80}",
                    f"{'Phase:':>28} {phase}",
                    f"{'Computation:':>28} {fps:.0f} steps/s ({iteration_time:.2f}s/iter)",
                    f"{'Collection time:':>28} {collect_time:.3f}s",
                    f"{'Learning time:':>28} {train_time:.3f}s",
                    f"{'World model loss:':>28} {float(wm_loss):.6f}",
                    f"{'World model rollout error:':>28} {float(wm_error):.6f}",
                ]
                if amp_metrics:
                    lines.extend([
                        f"{'AMP discriminator loss:':>28} {amp_metrics['train/amp_loss']:.6f}",
                        f"{'AMP gradient penalty:':>28} {amp_metrics['train/amp_grad_pen']:.6f}",
                        f"{'AMP D(policy):':>28} {amp_metrics['train/amp_policy_d']:.4f}",
                        f"{'AMP D(expert):':>28} {amp_metrics['train/amp_expert_d']:.4f}",
                    ])
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
                for tag, value in sorted(episode_metrics.items()):
                    lines.append(f"{tag + ':':>28} {value:.4f}")
                lines.append(f"{'Total simulation steps:':>28} {self.sim_steps}")
                lines.append(f"{'Total training time:':>28} {timedelta(seconds=int(self.wall_time))}")
                lines.append(f"{'ETA:':>28} {timedelta(seconds=int(remaining))}{eta_note}")
                print("\n".join(lines), flush=True)
                if summary_writer is not None:
                    summary = {
                        **amp_metrics,
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
                        "Loss/learning_rate": learning_rate,
                        "Loss/world_model": float(wm_loss),
                        "Loss/amp": amp_metrics.get("train/amp_loss"),
                        "Loss/amp_grad_pen": amp_metrics.get("train/amp_grad_pen"),
                        "Loss/amp_policy_pred": amp_metrics.get("train/amp_policy_d"),
                        "Loss/amp_expert_pred": amp_metrics.get("train/amp_expert_d"),
                        "Perf/total_fps": fps,
                        "Perf/collection_time": collect_time,
                        "Perf/learning_time": train_time,
                    }
                    if pi_loss is not None:
                        summary.update(
                            {
                                "train/pi_loss": pi_loss,
                                "train/v_loss": v_loss,
                                "train/imag_rew": imag_rew,
                                "train/pi_gradnorm": pi_gradnorm,
                                "train/imag_survival": imag_survival,
                                "Loss/policy": pi_loss,
                                "Loss/value_function": v_loss,
                            }
                        )
                    for tag, value in summary.items():
                        if value is not None:
                            summary_writer.add_scalar(tag, value, self.iteration)
                    for tag, value in episode_metrics.items():
                        summary_writer.add_scalar(tag, value, self.iteration)
                    if mean_episode_reward is not None:
                        summary_writer.add_scalar("Train/mean_reward/time", mean_episode_reward, int(self.wall_time))
                        summary_writer.add_scalar(
                            "Train/mean_episode_length/time", mean_episode_length, int(self.wall_time)
                        )

                last_metrics = {
                    "wm_loss": float(wm_loss),
                    "wm_err8": float(wm_error),
                    "sim_rew": mean_reward,
                    "pi_loss": pi_loss,
                    "v_loss": v_loss,
                    **amp_metrics,
                }
                self.stats = _empty_stats()
                self._episode_metrics.clear()

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
                    print(f"[amp-wm]: saved checkpoint {numbered}")

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


__all__ = ["AmpWmConfig", "AmpWmLearner"]
