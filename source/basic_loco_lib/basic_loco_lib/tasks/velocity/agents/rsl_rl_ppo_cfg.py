# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    # Explicitly route ``obs["policy"]`` to BOTH actor and critic networks so
    # we get true symmetric AC.  RSL-RL's auto-fallback when ``obs_groups`` is
    # ``{}`` (``rsl_rl/utils/utils.py:215-260``) inspects the env's obs dict
    # and quietly turns this into ASYMMETRIC AC if a ``"critic"`` key is
    # present (its ``default_set_name == "critic"`` branch picks the
    # ``"critic"`` group).  G1 Manager / Direct cfgs DO emit a privileged
    # ``critic`` obs group (with ``base_lin_vel`` and
    # ``enable_corruption=False``), so the fallback would silently activate
    # asymmetric AC and break a controlled symmetric-vs-asymmetric parity
    # study.  The explicit list below short-circuits that fallback.
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


