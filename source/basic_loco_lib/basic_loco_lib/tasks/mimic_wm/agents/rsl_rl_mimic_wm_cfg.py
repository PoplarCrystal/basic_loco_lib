"""World-model training settings; task observations and MLPs follow mimic."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MimicWmRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for the mimic-wm world-model learner."""

    # marker consumed by scripts/rsl_rl/train.py to select MimicWmRunner
    mimic_wm: bool = True

    ##
    # MimicWmConfig fields (see the learner dataclass for semantics).

    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    hist: int = 4
    # mimic-wm only tracks a few clips, so the policy matches the mimic PPO
    # nets' scale; the world model keeps Uni-SuperMimic's size.
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    policy_hidden: tuple[int, ...] = (512, 256, 128)  # mimic-scale policy
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2.0e-4
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    value_clip_min: float = -10.0
    value_clip_max: float = 1.0
    adaptive_replay: bool = False
    wm_uniform_mix: float = 0.2
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_backend: str = "tensorboard"
    amp: str = "bf16"
    compile: bool = True

    ##
    # rsl-rl runner interface (placeholders; the learner owns the networks).

    # unused by the WM path: the learner runs its own collect cadence (1/utd)
    num_steps_per_env = 1
    max_iterations = 320_000
    save_interval = 500
    experiment_name = ""  # same as task name
    load_checkpoint = "last.pt"
    seed = 0
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
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
