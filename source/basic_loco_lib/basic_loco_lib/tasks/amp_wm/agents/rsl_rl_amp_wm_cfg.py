"""Training settings for the AMP world-model task.

Self-contained: the runner config does not inherit from another task's config.
Everything here rides on the runner config dict; ``build_amp_wm_config`` filters
it down to ``AmpWmConfig`` fields, and the learner constructs the discriminator
from the ``amp_*`` entries.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from basic_loco_lib import PROJECT_DIR
from basic_loco_lib.tasks.amp_wm.diff.mdp import AMP_BODY_NAMES, ANCHOR_NAME


@configclass
class AmpWmRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for the AMP world-model learner."""

    # Marker consumed by scripts/rsl_rl/train.py and play.py.
    amp_wm: bool = True

    ##
    # World-model learner settings (mirrors mimic-wm's learner defaults).

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
    wm_statistics: str = "running"
    pi_batch: int = 4096
    lr: float = 2.0e-4
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    value_clip_min: float = -10.0
    value_clip_max: float = 1.0
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_backend: str = "tensorboard"
    amp: str = "none"
    compile: bool = True

    ##
    # AMP discriminator and expert data.

    amp_reward_coef: float = 0.1
    amp_task_reward_lerp: float = 0.75
    amp_motion_files: str = str(PROJECT_DIR / "data" / "amp")
    amp_discr_hidden_dims: tuple[int, ...] = (1024, 512, 256)
    amp_body_names: tuple[str, ...] = AMP_BODY_NAMES
    amp_anchor_name: str = ANCHOR_NAME
    amp_batch: int = 1024
    amp_updates: int = 1
    amp_lr: float = 1.0e-5
    amp_weight_decay_trunk: float = 1.0e-3
    amp_weight_decay_out: float = 0.1
    amp_grad_pen_lambda: float = 10.0

    ##
    # rsl-rl runner interface (placeholders; the learner owns the networks).

    num_steps_per_env = 1
    max_iterations = 100_001
    save_interval = 500
    experiment_name = "g1_amp_wm"
    load_checkpoint = "last.pt"
    # Seed 1 = the validated -8.5e-6 draw (get-up 24/24, best zero-cmd
    # behavior of the two verified seeds).
    seed = 1
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def g1_amp_wm_warp_runner_cfg() -> AmpWmRunnerCfg:
    """Keep Warp runs separate from standard runs when selecting checkpoints."""
    cfg = AmpWmRunnerCfg()
    cfg.experiment_name = "g1_amp_wm_warp"
    return cfg


__all__ = ["AmpWmRunnerCfg", "g1_amp_wm_warp_runner_cfg"]
