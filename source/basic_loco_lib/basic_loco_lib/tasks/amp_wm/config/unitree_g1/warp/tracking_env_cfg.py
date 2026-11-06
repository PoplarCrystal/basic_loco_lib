"""AMP locomotion task trained through a differentiable world model.

The environment side reuses AMP's Warp locomotion task (velocity command,
actor/critic observations, reward terms), with three task-level changes:

* observation groups are named ``policy`` / ``critic`` for the WM learner;
* the action term applies world-model execution noise;
* delayed recovery resets use the AMP task's events (no external pushes).

The policy is *not* trained by this reward config: the WM learner recomputes a
differentiable version of these terms in imagination (see ``diff/mdp.py``).
This config only drives the real-environment reward ``r`` stored in replay and
the episode-return logs.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_experimental.envs import mdp as warp_mdp
from isaaclab_experimental.managers import EventTermCfg as EventTerm
from isaaclab_experimental.managers import ObservationTermCfg as ObsTerm
from isaaclab_experimental.managers import RewardTermCfg as RewTerm
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.managers import TerminationTermCfg as DoneTerm
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from basic_loco_lib import PROJECT_DIR
from basic_loco_lib.robots.unitree_amp import (
    UNITREE_G1_29DOF_AMP_ACTION_SCALE,
)
from basic_loco_lib.robots.unitree_amp import UNITREE_G1_29DOF_AMP_CFG as ROBOT_CFG
from basic_loco_lib.tasks.amp_wm.mdp_warp import events as amp_events
from basic_loco_lib.tasks.amp_wm.mdp_warp import observations as amp_obs
from basic_loco_lib.tasks.amp_wm.mdp_warp import obs_history as amp_history
from basic_loco_lib.tasks.amp_wm.mdp_warp import rewards as amp_rewards
from basic_loco_lib.tasks.amp_wm.mdp_warp import terminations as amp_terminations
from basic_loco_lib.tasks.amp_wm.mdp_warp.commands import WarpUniformVelocityCommand
from basic_loco_lib.tasks.amp_wm.diff.mdp import AMP_BODY_NAMES, ANCHOR_NAME
from basic_loco_lib.tasks.amp_wm.mdp_warp.actions import AmpWmJointPositionAction

##
# Scene
##


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the AMP locomotion scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# Commands
##


@configclass
class CommandsCfg:
    twist = UniformVelocityCommandCfg(
        class_type=WarpUniformVelocityCommand,
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.25,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-3.14 / 2, 3.14 / 2),
            heading=(-math.pi / 2, math.pi / 2),
        ),
    )


##
# Actions
##


@configclass
class ActionsCfg:
    JointPositionAction = warp_mdp.JointPositionActionCfg(
        class_type=AmpWmJointPositionAction,
        asset_name="robot",
        joint_names=[".*"],
        scale=UNITREE_G1_29DOF_AMP_ACTION_SCALE,
        use_default_offset=True,
    )


##
# Observations
##

_AMP_BODY_NAMES = AMP_BODY_NAMES
_ANCHOR_NAME = ANCHOR_NAME


@configclass
class ObservationsCfg:
    """Observation specifications for the world-model AMP task."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Term order mirrors the AMP task's actor group.
        command = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.generated_commands, 4, frame_dim=3),
            params={"command_name": "twist"},
            history_length=0,
        )
        projected_gravity = ObsTerm(
            func=amp_history.make_history_obs(
                warp_mdp.projected_gravity, 4, frame_dim=3, noise=Unoise(n_min=-0.05, n_max=0.05)
            ),
            history_length=0,
        )
        base_ang_vel = ObsTerm(
            func=amp_history.make_history_obs(
                warp_mdp.base_ang_vel, 4, frame_dim=3, noise=Unoise(n_min=-0.2, n_max=0.2)
            ),
            history_length=0,
        )
        joint_pos = ObsTerm(
            func=amp_history.make_history_obs(
                warp_mdp.joint_pos_rel, 4, frame_dim=29, noise=Unoise(n_min=-0.01, n_max=0.01)
            ),
            history_length=0,
        )
        joint_vel = ObsTerm(
            func=amp_history.make_history_obs(
                warp_mdp.joint_vel_rel, 4, frame_dim=29, noise=Unoise(n_min=-0.5, n_max=0.5)
            ),
            history_length=0,
        )
        actions = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.last_action, 4, frame_dim=29),
            history_length=0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # Term order mirrors the AMP task's critic group.
        command = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.generated_commands, 4, frame_dim=3),
            params={"command_name": "twist"},
            history_length=0,
        )
        projected_gravity = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.projected_gravity, 4, frame_dim=3),
            history_length=0,
        )
        base_lin_vel = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.base_lin_vel, 4, frame_dim=3),
            history_length=0,
        )
        base_ang_vel = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.base_ang_vel, 4, frame_dim=3),
            history_length=0,
        )
        joint_pos = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.joint_pos_rel, 4, frame_dim=29),
            history_length=0,
        )
        joint_vel = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.joint_vel_rel, 4, frame_dim=29),
            history_length=0,
        )
        actions = ObsTerm(
            func=amp_history.make_history_obs(warp_mdp.last_action, 4, frame_dim=29),
            history_length=0,
        )
        body_pos_b = ObsTerm(
            func=amp_history.make_history_obs(amp_obs.robot_body_pos_b, 4, frame_dim=39),
            history_length=0,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
                "body_cfg": SceneEntityCfg(
                    "robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True
                ),
            },
        )
        body_ori_b = ObsTerm(
            func=amp_history.make_history_obs(amp_obs.robot_body_ori_b, 4, frame_dim=78),
            history_length=0,
            params={
                "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
                "body_cfg": SceneEntityCfg(
                    "robot", body_names=list(_AMP_BODY_NAMES), preserve_order=True
                ),
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Events
##

_MOTION_BASE = PROJECT_DIR / "data" / "amp"
_MOTION_DIR = str(_MOTION_BASE / "WalkandRun")
_RECOVERY_DIR = str(_MOTION_BASE / "Recovery")


@configclass
class EventCfg:
    """Startup/reset events; delayed recovery mirrors the AMP task."""

    init_motion_loader = EventTerm(
        func=amp_events.init_motion_loader_warp,
        mode="startup",
        params={
            "motion_dir": _MOTION_DIR,
            "recovery_dir": _RECOVERY_DIR,
            # Delay envs stay fallen, then reset into Recovery clips (AMP's
            # recovery recipe) so the policy sees get-up states.
            "delay_reset_env_ratio": 0.4,
            "max_delay_steps": 250,
            "max_delay_joint_velocity": 50.0,
        },
    )
    reset_from_motion = EventTerm(
        func=amp_events.reset_from_motion_data_warp,
        mode="reset",
        params={
            "motion_dir": _MOTION_DIR,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )


##
# Rewards (real environment; the learner recomputes a differentiable version)
##


@configclass
class RewardsCfg:
    track_anchor_lin_vel = RewTerm(
        func=amp_rewards.track_anchor_linear_velocity,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 1.0,
            "mask_delay": True,
            "delay_env_rew_ratio": 0.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        },
    )
    track_anchor_ang_vel = RewTerm(
        func=amp_rewards.track_anchor_angular_velocity,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 3.14,
            "mask_delay": True,
            "delay_env_rew_ratio": 0.0,
            "anchor_cfg": SceneEntityCfg("robot", body_names=_ANCHOR_NAME),
        },
    )
    track_root_height = RewTerm(
        func=amp_rewards.track_root_height,
        weight=1.0,
        # AMP's delay semantics: height is a recovery-only reward (3.5x while
        # the robot is fallen, zero otherwise).
        params={"std": 0.3, "mask_delay": True, "delay_env_rew_ratio": 3.5},
    )
    body_ang_vel_xy = RewTerm(
        func=amp_rewards.body_ang_vel_xy_l2,
        weight=0.5,
        params={
            "std": 3.14,
            "mask_delay": True,
            "delay_env_rew_ratio": 0.0,
            "body_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        },
    )
    # The world model reproduces these terms: ``joint_acc`` uses the sim's
    # instantaneous acceleration here and the WM state's joint_vel finite
    # difference in ``diff/mdp.py``; ``is_terminated`` is mirrored by the
    # learner's ``AmpDiffMDP.fall_penalty``.
    joint_acc = RewTerm(func=warp_mdp.joint_acc_l2, weight=-2.5e-7)
    joint_limit = RewTerm(func=warp_mdp.joint_pos_limits, weight=-10.0)
    action_rate = RewTerm(func=warp_mdp.action_rate_l2, weight=-0.01)
    is_terminated = RewTerm(func=warp_mdp.is_terminated, weight=-200.0)


##
# Terminations
##


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=warp_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=amp_terminations.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    bad_root_height = DoneTerm(
        func=warp_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    )


##
# Environment configuration
##


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """G1 AMP locomotion task trained through a differentiable world model."""

    sim: SimulationCfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                # Suppress line-search limit warnings; keep other warnings enabled.
                class_type="basic_loco_lib.utils.physics.mjwarp_manager:NewtonMJWarpQuietLSManager",
                # Extra contact capacity for capsule feet and self-collisions.
                njmax=256,
                nconmax=64,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                ls_parallel=True,
                ls_iterations=15,
            ),
            num_substeps=1,
            debug_mode=False,
        ),
        use_newton_actuators=False,
    )
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    # Applied only to the world-model data collection action term.
    action_noise: float = 0.1

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.action_noise = 0.0
