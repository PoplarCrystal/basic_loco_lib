"""World model and AMP-observation MLP actor/critic."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .contracts import State

from .geometry import invariant_features, matmul, matvec, so3_exp, so3_log, yaw_mat


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: type[nn.Module] = nn.ELU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden_dim in hidden:
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running moments, then slow EMA adaptation after the initial fit."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.mean: torch.Tensor
        self.var: torch.Tensor
        self.count: torch.Tensor
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            # Keep adapting across resumed recovery/noise curricula without
            # abruptly resetting the statistics learned by the world model.
            rate = 1e-3
            batch_mean = x.mean(0)
            batch_var = x.var(0, unbiased=False)
            delta = batch_mean - self.mean
            self.mean.add_(rate * delta)
            self.var.copy_((1-rate) * self.var + rate * batch_var + rate * (1-rate) * delta.square())
            return
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        ) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run an MLP under optional autocast, returning fp32 for state arithmetic."""
    if AMP_DTYPE is None:
        return net(x)
    device_type = x.device.type
    if device_type not in ("cuda", "mps"):
        return net(x)
    with torch.autocast(device_type, dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Predict frame-local state increments and integrate them in fp32."""

    def __init__(
        self,
        hidden: Sequence[int] = (2048, 2048, 2048),
        num_joints: int = 29,
        context_dim: int = 33,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.context_dim = context_dim
        self.in_dim = 13 + 3 * num_joints + context_dim
        self.out_dim = 12 + 2 * num_joints
        self.net = mlp(self.in_dim, hidden, self.out_dim, act=nn.SiLU, layer_norm=True)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(
        self, state: State, action: torch.Tensor, normalized_context: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((invariant_features(state), action, normalized_context), -1)

    @staticmethod
    def targets(state: State, next_state: State) -> torch.Tensor:
        """Return frame-local increments from ``state`` to ``next_state``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        next_position, next_rotation, next_lin_vel, next_ang_vel, next_joint_pos, next_joint_vel = (
            next_state
        )
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                matvec(heading.transpose(-1, -2), next_position - position),
                so3_log(matmul(rotation_t, next_rotation)),
                matvec(rotation_t, next_lin_vel - lin_vel),
                matvec(rotation_t, next_ang_vel - ang_vel),
                next_joint_pos - joint_pos,
                next_joint_vel - joint_vel,
            ),
            -1,
        )

    def step(
        self,
        state: State,
        action: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> State:
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        prediction = run_mlp(
            self.net,
            self.in_norm(self.inputs(state, action, normalized_context)),
        )
        increment = prediction * self.out_norm.std() + self.out_norm.mean
        heading = yaw_mat(rotation)
        d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel = increment.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), -1
        )
        next_position = position + matvec(heading, d_position)
        next_rotation = matmul(rotation, so3_exp(d_rotation))
        next_lin_vel = lin_vel + matvec(rotation, d_lin_vel)
        next_ang_vel = ang_vel + matvec(rotation, d_ang_vel)
        return (
            next_position,
            next_rotation,
            next_lin_vel,
            next_ang_vel,
            joint_pos + d_joint_pos,
            joint_vel + d_joint_vel,
        )

    @staticmethod
    def state_error(predicted_state: State, state: State) -> torch.Tensor:
        """Return frame-local errors in the same layout as ``targets``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        (
            predicted_position,
            predicted_rotation,
            predicted_lin_vel,
            predicted_ang_vel,
            predicted_joint_pos,
            predicted_joint_vel,
        ) = predicted_state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                matvec(heading.transpose(-1, -2), predicted_position - position),
                so3_log(matmul(rotation_t, predicted_rotation)),
                matvec(rotation_t, predicted_lin_vel - lin_vel),
                matvec(rotation_t, predicted_ang_vel - ang_vel),
                predicted_joint_pos - joint_pos,
                predicted_joint_vel - joint_vel,
            ),
            -1,
        )


class Policy(nn.Module):
    """Deterministic MLP consuming the AMP policy observation."""

    def __init__(self, obs_dim: int, num_joints: int = 29, hidden=(512, 256, 128)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, num_joints)
        self.obs_norm = RunningNorm(obs_dim)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.obs_norm(obs))

    @torch.no_grad()
    def update_normalization(self, obs):
        _update_input_norm(self.net[0], self.obs_norm, obs)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _upgrade_identity_norm(state_dict, prefix + "obs_norm.", self.obs_norm)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(x)).squeeze(-1)

    @torch.no_grad()
    def update_normalization(self, obs):
        _update_input_norm(self.net[0], self.norm, obs)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _upgrade_identity_norm(state_dict, prefix + "norm.", self.norm)
        key = prefix + "net.0.weight"
        if key in state_dict and state_dict[key].shape[1] == self.net[0].in_features - 2:
            state_dict[key] = torch.nn.functional.pad(state_dict[key], (0, 2))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def _upgrade_identity_norm(state_dict, prefix, normalizer):
    """Old actor/critic checkpoints had an Identity input normalizer."""
    if prefix + "mean" not in state_dict:
        state_dict[prefix + "mean"] = torch.zeros_like(normalizer.mean)
        state_dict[prefix + "var"] = torch.ones_like(normalizer.var) - normalizer.eps**2
        state_dict[prefix + "count"] = torch.zeros_like(normalizer.count)


@torch.no_grad()
def _update_input_norm(first_layer, normalizer, obs):
    """Change coordinates without changing the represented actor/value map."""
    old_mean, old_std = normalizer.mean.clone(), normalizer.std().clone()
    normalizer.update(obs)
    shift = (normalizer.mean - old_mean) / old_std
    first_layer.bias.add_(first_layer.weight @ shift)
    first_layer.weight.mul_((normalizer.std() / old_std).unsqueeze(0))


__all__ = ["AMP_DTYPE", "Critic", "Policy", "RunningNorm", "WorldModel", "mlp", "run_mlp"]
