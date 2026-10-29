"""RL configuration for standard and Warp Unitree G1 AMP locomotion environments."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from basic_loco_lib import PROJECT_DIR


@configclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Extended runner config with AMP-specific parameters (v5 rsl-rl-lib format)."""

    # Override algorithm to use AMPPPO
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
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
        class_name="basic_loco_lib.algo.amp.algorithms.amp_ppo:AMPPPO",
    )

    # Actor model
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    # Critic model
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    num_steps_per_env = 24
    max_iterations = 100001
    save_interval = 100
    experiment_name = "g1_amp_locomotion"
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}

    # AMP-specific parameters
    amp_reward_coef: float = 0.1
    amp_motion_files: str = ""
    amp_num_preload_transitions: int = 200000
    amp_task_reward_lerp: float = 0.75
    amp_discr_hidden_dims: list = (1024, 512, 256)
    min_normalized_std: list = (0.05,) * 29
    amp_body_names: tuple = ()
    amp_anchor_name: str = ""
    amp_replay_buffer_size: int = 1000000
    check_for_nan: bool = False
    detailed_nan_check: bool = False


def g1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """Create RL runner configuration for Unitree G1 AMP locomotion task."""
    return RslRlAmpRunnerCfg(
        amp_motion_files=str(PROJECT_DIR / "data" / "amp"),
        amp_body_names=(
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        amp_anchor_name="torso_link",
    )


def g1_amp_warp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """Keep Warp runs separate from standard runs when selecting checkpoints."""
    cfg = g1_amp_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_locomotion_warp"
    return cfg
