"""Fixed-size device-resident replay laid out as ``(time, environment)``.

Ported from Uni-SuperMimic ``rl/algos/supermimic/replay.py``; the buffer
semantics (window masking, boundary-padded histories) are unchanged.
"""

from __future__ import annotations

import torch


def _make_fields(
    num_joints: int,
    context_dim: int,
) -> dict[str, tuple[int | tuple[()], torch.dtype]]:
    state_dim = 13 + 2 * num_joints
    return {
        "s": (state_dim, torch.float32),
        "s1": (state_dim, torch.float32),
        "a": (num_joints, torch.float32),
        "a_prev": (num_joints, torch.float32),
        "xi": (context_dim, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
        "episode_done": ((), torch.bool),
        "term": ((), torch.bool),
        "contact": (2, torch.float32),
        "a_clean": (num_joints, torch.float32),
        "xi1": (context_dim, torch.float32),
        "recovery": (3, torch.float32),
        "recovery1": (3, torch.float32),
    }


class ReplayBuffer:
    def __init__(
        self,
        T: int,
        N: int,
        device: str | torch.device,
        num_joints: int = 29,
        context_dim: int = 33,
    ):
        self.T = T
        self.N = N
        self.device = torch.device(device)
        self.fields = _make_fields(num_joints, context_dim)
        self.data = {}
        for key, (shape, dtype) in self.fields.items():
            field_shape = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[key] = torch.zeros(field_shape, dtype=dtype, device=self.device)
        self.ptr = 0
        self.size = 0

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        for key in self.fields:
            self.data[key][self.ptr] = transition[key]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time indices (zero is oldest) to storage rows."""
        return (self.ptr - self.size + age) % self.T

    def _sample_cells(self, B: int, ages: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``B`` (age, env) cells."""
        age = torch.randint(0, ages, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        return age, env

    def sample_windows(self, B: int, k: int) -> dict[str, torch.Tensor]:
        """Sample ``k``-step windows and mask transitions after episode ends."""
        if not 1 <= k <= self.size:
            raise ValueError(f"window length must be in [1, {self.size}], got {k}")
        start, env = self._sample_cells(B, self.size - k + 1)
        ages = start.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        envs = env.unsqueeze(-1).expand_as(rows)
        out = {
            key: self.data[key][rows, envs]
            for key in ("s", "s1", "a", "xi", "done", "episode_done", "contact")
        }
        boundary = (out["done"] | out["episode_done"]).float()
        mask = torch.cumprod(
            torch.cat((torch.ones(B, 1, device=self.device), 1 - boundary[:, :-1]), -1),
            -1,
        )
        out["mask"] = mask
        return out

    def sample_starts(
        self,
        B: int,
        hist: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample imagination starts and oldest-first, boundary-padded histories."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if hist < 1:
            raise ValueError(f"history length must be positive, got {hist}")
        age, env = self._sample_cells(B, self.size)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        valid = (back.unsqueeze(0) <= ep_step.unsqueeze(-1)) & (back.unsqueeze(0) <= age.unsqueeze(-1))
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        history_rows = self._row(age.unsqueeze(-1) - back)
        history_envs = env.unsqueeze(-1).expand_as(history_rows)
        start = {key: value[row, env] for key, value in self.data.items()}
        history = {
            key: self.data[key][history_rows, history_envs] for key in ("s", "a_prev", "xi", "f")
        }
        history["valid"] = valid
        history["complete"] = (ep_step <= age) | (age >= hist - 1)
        return start, history


__all__ = ["ReplayBuffer"]
