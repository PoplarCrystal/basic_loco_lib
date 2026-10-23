"""Warp counterpart of the standard orientation termination."""
import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg


@wp.kernel
def _bad_orientation(pose: wp.array(dtype=wp.transformf), gravity_w: wp.array(dtype=wp.vec3f),
                     limit: float, out: wp.array(dtype=bool)):
    i = wp.tid()
    gravity = wp.quat_rotate_inv(wp.transform_get_rotation(pose[i]), gravity_w[i])
    out[i] = wp.abs(wp.acos(-gravity[2])) > limit


def bad_orientation(env, out, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')):
    data = env.scene[asset_cfg.name].data
    wp.launch(_bad_orientation, dim=env.num_envs,
              inputs=[data.root_link_pose_w.warp, data.GRAVITY_VEC_W.warp, limit_angle, out], device=env.device)
