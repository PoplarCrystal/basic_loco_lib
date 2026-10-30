# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-aware event term wrappers for the basic_loco_lib tasks.

Provides per-shape Newton material randomization using the
``shape_material_mu`` / ``shape_material_restitution`` bindings on
:class:`newton.selection.ArticulationView`.  Drops in for
:func:`isaaclab.envs.mdp.events.randomize_rigid_body_material` when the
PhysX-only impl in the active IsaacLab release does not yet ship the
Newton auto-dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def newton_randomize_rigid_body_material(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | slice | None,
    static_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    asset_cfg: "SceneEntityCfg",
    dynamic_friction_range: tuple[float, float] | None = None,
    num_buckets: int | None = None,
    make_consistent: bool = False,
) -> None:
    """Per-shape friction (``mu``) + restitution randomization for Newton.

    Samples each ``(env, shape)`` pair independently from the given
    ``Uniform`` ranges, writes them into Newton's per-shape material
    bindings, and notifies the solver to re-propagate the change to the
    contact-resolution buffers (``mjw_model.geom_friction``).

    Used by ``EventTerm(mode="startup")`` so this fires once during
    ``sim.reset()`` (PHYSICS_READY callback) and the per-shape values
    persist for the rest of the simulation.

    PhysX-only parameters (``dynamic_friction_range``, ``num_buckets``,
    ``make_consistent``) are accepted but ignored — kept so the cfg can
    drop in either this function or
    :func:`isaaclab.envs.mdp.randomize_rigid_body_material` without
    rewriting ``params``.

    Args:
        env: Environment instance providing ``scene`` and ``device``.
        env_ids: Index tensor or slice selecting envs, or ``None`` for all envs.
        static_friction_range: ``Uniform(low, high)`` for ``mu`` [unitless].
        restitution_range: ``Uniform(low, high)`` for restitution [unitless].
        asset_cfg: Selects the asset whose shape materials get randomized.
        dynamic_friction_range: PhysX-only, ignored.
        num_buckets: PhysX-only bucket count, ignored.
        make_consistent: PhysX-only static<->dynamic clamp flag, ignored.
    """
    del dynamic_friction_range, num_buckets, make_consistent  # PhysX-only, ignored by Newton

    import isaaclab_newton.physics.newton_manager as newton_manager_module
    from newton import ModelFlags

    asset = env.scene[asset_cfg.name]
    device = env.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)[env_ids]
    else:
        env_ids = env_ids.to(device).long()

    model = newton_manager_module.NewtonManager.get_model()
    friction_binding = asset._root_view.get_attribute("shape_material_mu", model)[:, 0]
    restitution_binding = asset._root_view.get_attribute("shape_material_restitution", model)[:, 0]

    # Mirror PhysX ``multiply`` friction combine semantics:
    # ``μ_eff = μ_robot × μ_ground``.
    #
    # MuJoCo does not have a ``multiply`` combine mode — it weighted-averages
    # via ``geom_solmix``.  We approximate ``multiply`` by:
    #   1. Scaling the per-shape μ sample by ``μ_ground``                (this line below)
    #   2. Setting ``ground.solmix = 0`` so it is transparent in average  (further below)
    # Together: Newton ``μ_eff`` per (env, shape) ≈ PhysX ``μ_robot × μ_ground``.
    #
    # ``μ_ground`` is read from ``self.scene.terrain.physics_material.static_friction``.
    # Falls back to 1.0 if the cfg field is absent (e.g. test envs).
    try:
        ground_mu = float(env.scene.cfg.terrain.physics_material.static_friction)
    except (AttributeError, TypeError):
        ground_mu = 1.0

    n_shapes = friction_binding.shape[1]
    fr_lo, fr_hi = static_friction_range
    fr_lo_scaled = fr_lo * ground_mu
    fr_hi_scaled = fr_hi * ground_mu
    re_lo, re_hi = restitution_range
    friction_samples = math_utils.sample_uniform(
        fr_lo_scaled, fr_hi_scaled, (len(env_ids), n_shapes), device=device
    )
    restitution_samples = math_utils.sample_uniform(
        re_lo, re_hi, (len(env_ids), n_shapes), device=device
    )

    friction_view = wp.to_torch(friction_binding)
    restitution_view = wp.to_torch(restitution_binding)
    shape_idx = torch.arange(n_shapes, dtype=torch.long, device=device)
    friction_view[env_ids[:, None], shape_idx] = friction_samples
    restitution_view[env_ids[:, None], shape_idx] = restitution_samples

    # Step 2 of the PhysX-multiply emulation: make the ground transparent in
    # MuJoCo friction average by setting its solmix to 0.  Together with the
    # μ-scaling above, ``μ_eff = (0 * μ_g + 1 * μ_robot_scaled) / (0 + 1) =
    # μ_robot * μ_ground`` — matching PhysX ``multiply`` for any ``μ_ground``.
    # Assumes the ground is shape index 0 (true for ``TerrainImporterCfg``
    # which inserts the ground prim before any robot prims).
    if hasattr(model, "mujoco") and getattr(model.mujoco, "geom_solmix", None) is not None:
        wp.to_torch(model.mujoco.geom_solmix)[0] = 0.0

    newton_manager_module.NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
