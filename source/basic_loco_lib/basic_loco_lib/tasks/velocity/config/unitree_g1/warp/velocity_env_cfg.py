"""Newton/Warp velocity task sharing all training parameters with standard."""
from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from isaaclab_experimental.envs.frontend import WarpFrontend
from isaaclab_tasks_experimental.core.velocity.mdp.curriculums import terrain_levels_vel

from basic_loco_lib.tasks.velocity import mdp_warp
from basic_loco_lib.tasks.velocity.config.unitree_g1.standard.velocity_env_cfg import RobotEnvCfg as StandardEnvCfg
from basic_loco_lib.tasks.velocity.mdp_warp.curriculums import lin_vel_cmd_levels
from basic_loco_lib.tasks.velocity.mdp_warp.obs_history import make_history_obs


@configclass
class RobotEnvCfg(StandardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = self.sim.physics.newton
        # The shared generator currently consists of zero-volume plane meshes.
        # Use IsaacLab's native Newton heightfield conversion for that surface.
        for terrain in self.scene.terrain.terrain_generator.sub_terrains.values():
            if isinstance(terrain, MeshPlaneTerrainCfg):
                terrain.convert_to_heightfield = True
        # Resolve official Warp twins; command and curriculum managers retain
        # their supported Torch implementations.
        for group in (self.actions, self.events, self.rewards, self.terminations):
            for term in vars(group).values():
                if hasattr(term, 'func'):
                    term.func = getattr(mdp_warp, term.func.__name__)
        self.curriculum.terrain_levels.func = terrain_levels_vel
        self.curriculum.lin_vel_cmd_levels.func = lin_vel_cmd_levels
        dims = {'base_lin_vel': 3, 'base_ang_vel': 3, 'projected_gravity': 3,
                'velocity_commands': 3, 'joint_pos_rel': 29, 'joint_vel_rel': 29, 'last_action': 29}
        for group in (self.observations.policy, self.observations.critic):
            history = group.history_length
            for name, term in vars(group).items():
                if not hasattr(term, 'func'):
                    continue
                source = getattr(mdp_warp, term.func.__name__)
                term.func = make_history_obs(source, history, frame_dim=dims[name], noise=term.noise)
                term.noise = None
                term.history_length = 0
            group.history_length = 0
        WarpFrontend.adapt_cfg(self)


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotFlatEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = 'plane'
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None


@configclass
class RobotFlatPlayEnvCfg(RobotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
