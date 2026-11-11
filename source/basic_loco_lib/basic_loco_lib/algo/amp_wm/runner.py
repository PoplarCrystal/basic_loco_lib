"""rsl-rl style runner bridging ``train.py`` to the AMP-WM learner.

Self-contained: no other task's runner is imported. Inference skips the
discriminator and expert motion data (``log_dir`` is ``None`` in play mode).
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import torch

from .learner import AmpWmConfig, AmpWmLearner

# Agent-config keys consumed by the learner; anything else in the dict is
# ignored (class_name, experiment_name, and standard rsl-rl runner fields).
_LEARNER_FIELDS = {field.name for field in dataclasses.fields(AmpWmConfig)}
_TUPLE_FIELDS = {
    "wm_hidden",
    "policy_hidden",
    "critic_hidden",
    "test_hidden_dims",
    "amp_discr_hidden_dims",
    "amp_body_names",
}


def build_amp_wm_config(train_cfg: dict[str, Any]) -> AmpWmConfig:
    """Filter a configclass-derived dict down to ``AmpWmConfig`` fields."""
    kwargs: dict[str, Any] = {}
    for name in _LEARNER_FIELDS:
        if name in train_cfg and train_cfg[name] is not None:
            value = train_cfg[name]
            if name in _TUPLE_FIELDS and isinstance(value, (list, tuple)):
                value = tuple(value)
            kwargs[name] = value
    return AmpWmConfig(**kwargs)


class AmpWmRunner:
    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        self.env = env
        self._training = log_dir is not None
        cfg = build_amp_wm_config(train_cfg)
        self.learner = AmpWmLearner(
            env,
            cfg,
            log_dir or "./logs/amp_wm",
            device=device,
            amp_enabled=self._training,
        )

    @property
    def cfg(self) -> AmpWmConfig:
        return self.learner.cfg

    def add_git_repo_to_log(self, *args: Any, **kwargs: Any) -> None:
        """No-op retained for interface parity with rsl-rl runners."""

    def learn(
        self,
        num_learning_iterations: int | None = None,
        init_at_random_ep_len: bool = False,  # noqa: ARG002 (interface parity)
        **kwargs: Any,
    ) -> dict[str, float | int | bool]:
        if num_learning_iterations is not None:
            self.learner.cfg.iterations = num_learning_iterations
        return self.learner.train()

    def save(self, path: str | Path | None = None, include_optimizers: bool = True) -> Path:
        return self.learner.save(path, include_optimizers=include_optimizers)

    def load(self, path: str | Path | None = None) -> None:
        if path is None:
            raise ValueError("AmpWmRunner.load requires an explicit checkpoint path")
        self.learner.load(path, load_optimizers=self._training)

    def _export_sample_inputs(self):
        return torch.zeros(
            1, self.learner.mdp.obs_dim * self.learner.cfg.hist, device="cpu"
        )

    def export_policy_to_jit(self, path, filename="policy.pt", **kwargs):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        # CPU export bypasses training autocast and keeps deployment operators FP32.
        policy = copy.deepcopy(self.learner.pi).cpu().float().eval()
        with torch.no_grad():
            traced = torch.jit.trace(policy, self._export_sample_inputs())
        traced.save(str(destination / filename))

    def export_policy_to_onnx(self, path, filename="policy.onnx", **kwargs):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        policy = copy.deepcopy(self.learner.pi).cpu().float().eval()
        torch.onnx.export(
            policy, self._export_sample_inputs(), str(destination / filename),
            input_names=["obs"], output_names=["actions"], opset_version=17,
        )

    def eval_mode(self) -> None:
        self.learner.pi.eval()
        self.learner.wm.eval()

    def train_mode(self) -> None:
        self.learner.pi.train()
        self.learner.wm.train()

    def get_inference_policy(self, device=None):
        return AmpWmInferencePolicy(self.learner, device)


class AmpWmInferencePolicy:
    """Consume the task's policy observation, without an extra goal or history."""

    def __init__(self, learner, device=None):
        self.learner = learner
        self.device = device

    def __call__(self, obs):
        if not torch.is_tensor(obs):
            obs = obs["policy"]
        with torch.no_grad():
            action = self.learner.pi(obs.to(self.learner.device))
        return action.to(self.device or self.learner.env_device)

    def reset(self, dones=None):
        pass  # Observation managers own the history and reset masks.


__all__ = ["AmpWmInferencePolicy", "AmpWmRunner", "build_amp_wm_config"]
