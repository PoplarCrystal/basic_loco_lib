"""Injected task contract for AMP world-model training.

The task owns the differentiable analytic heads (observation / reward / soft
termination), the AMP discriminator features and the velocity command that
rides in the world-model context.
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

AMP_WM_TRANSITION_KEY = "amp_wm"
AMP_WM_STATE_KEY = "amp_wm_state"


@runtime_checkable
class AmpWmMdpProtocol(Protocol):
    """Differentiable analytic heads and command context owned by a task."""

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

    def to(self, device: str | torch.device) -> AmpWmMdpProtocol:
        """Return a copy on ``device`` for learner-side use off the env device."""
        ...


@runtime_checkable
class SupportsAmpWmProtocol(Protocol):
    """Environment exposing its task-owned AMP world-model implementation."""

    @property
    def amp_wm_mdp(self) -> AmpWmMdpProtocol: ...


@runtime_checkable
class SupportsAmpWmSamplerProtocol(Protocol):
    """Optional environment capability: an adaptive start-segment sampler."""

    @property
    def amp_wm_sampler(self) -> Any: ...


def get_amp_wm_mdp(env: Any) -> AmpWmMdpProtocol:
    """Return the injected task MDP, failing closed for unsupported envs."""
    if not isinstance(env, SupportsAmpWmProtocol):
        raise TypeError(f"{type(env).__name__} does not provide the amp-wm MDP contract")
    return env.amp_wm_mdp


__all__ = [
    "AMP_WM_STATE_KEY",
    "AMP_WM_TRANSITION_KEY",
    "AmpWmMdpProtocol",
    "State",
    "SupportsAmpWmProtocol",
    "SupportsAmpWmSamplerProtocol",
    "get_amp_wm_mdp",
]
