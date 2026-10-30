"""Injected task contract for mimic-wm world-model training.

Ported from Uni-SuperMimic ``rl/contracts/supermimic.py``; the task owns the
differentiable analytic heads (obs / reward / soft termination) and the
reference motion data, and exposes them to the learner through this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

State = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

MIMIC_WM_TRANSITION_KEY = "mimic_wm"
MIMIC_WM_STATE_KEY = "mimic_wm_state"


@runtime_checkable
class MimicWmMdpProtocol(Protocol):
    """Differentiable analytic heads and reference data owned by a task."""

    num_joints: int
    context_dim: int
    obs_dim: int
    critic_dim: int
    actor_dims: tuple[int, ...]
    critic_dims: tuple[int, ...]
    max_step_reward: float

    def obs(
        self,
        state: State,
        previous_action: torch.Tensor,
        context: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor: ...

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor: ...

    def reward(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor: ...

    def soft_termination(self, state: State, frames: torch.Tensor) -> torch.Tensor: ...

    def critic_features(
        self,
        state: State,
        frames: torch.Tensor,
        context: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor: ...

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor: ...

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor: ...

    def to(self, device: str | torch.device) -> MimicWmMdpProtocol:
        """Return a copy on ``device`` for learner-side use off the env device."""
        ...


@runtime_checkable
class SupportsMimicWmProtocol(Protocol):
    """Environment exposing its task-owned mimic-wm implementation."""

    @property
    def mimic_wm_mdp(self) -> MimicWmMdpProtocol: ...


@runtime_checkable
class SupportsMimicWmSamplerProtocol(Protocol):
    """Optional environment capability: an adaptive start-segment sampler.

    Exposed by task envs whose command tracks per-segment mastery. The learner
    duck-types this to weight replay sampling and to checkpoint sampler state.
    """

    @property
    def mimic_wm_sampler(self) -> Any: ...


def get_mimic_wm_mdp(env: Any) -> MimicWmMdpProtocol:
    """Return the injected task MDP, failing closed for unsupported envs."""
    if not isinstance(env, SupportsMimicWmProtocol):
        raise TypeError(f"{type(env).__name__} does not provide the mimic-wm MDP contract")
    return env.mimic_wm_mdp


__all__ = [
    "MIMIC_WM_STATE_KEY",
    "MIMIC_WM_TRANSITION_KEY",
    "State",
    "SupportsMimicWmSamplerProtocol",
    "SupportsMimicWmProtocol",
    "MimicWmMdpProtocol",
    "get_mimic_wm_mdp",
]
