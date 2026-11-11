# Copyright (c) 2025-2026, The IsaacNewton Project Developers.
# Running mean/std normalizer for AMP observations.

import numpy as np
import torch


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device="cpu"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    def update(self, arr):
        device = arr.device if isinstance(arr, torch.Tensor) else (self.mean.device if isinstance(self.mean, torch.Tensor) else "cpu")
        self._move_to(device)
        arr = torch.as_tensor(arr, dtype=torch.float32, device=device)
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0, unbiased=False)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        device = batch_mean.device if isinstance(batch_mean, torch.Tensor) else (self.mean.device if isinstance(self.mean, torch.Tensor) else "cpu")
        self._move_to(device)
        batch_mean = torch.as_tensor(batch_mean, dtype=self.mean.dtype, device=self.mean.device)
        batch_var = torch.as_tensor(batch_var, dtype=self.var.dtype, device=self.var.device)
        batch_count = torch.as_tensor(batch_count, dtype=self.count.dtype, device=self.count.device)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / tot_count
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def _move_to(self, device):
        self.mean = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        self.var = torch.as_tensor(self.var, dtype=torch.float32, device=device)
        self.count = torch.as_tensor(self.count, dtype=torch.float32, device=device)


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0, device="cpu"):
        super().__init__(shape=input_dim, device=device)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        mean = self.mean.detach().cpu().numpy()
        var = self.var.detach().cpu().numpy()
        return np.clip((input - mean) / np.sqrt(var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean = torch.as_tensor(self.mean, device=device, dtype=input.dtype)
        std = torch.sqrt(torch.as_tensor(self.var, device=device, dtype=input.dtype) + self.epsilon)
        return torch.clamp((input - mean) / std, -self.clip_obs, self.clip_obs)
