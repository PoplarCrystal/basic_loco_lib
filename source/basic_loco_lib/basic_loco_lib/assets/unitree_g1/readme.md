# Unitree G1 asset sources

## USD

The `usd/` assets come from the official Unitree model dataset:

- Dataset: https://huggingface.co/datasets/unitreerobotics/unitree_model
- G1 29-DoF assets: https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main/G1/29dof/usd
- Model: `g1_29dof_rev_1_0.usd` and its configuration layers.

## URDF and meshes

The `urdf/g1_29dof_rev_1_0.urdf` model comes from the official Unitree ROS repository:

- Repository: https://github.com/unitreerobotics/unitree_ros
- Source: https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof_rev_1_0.urdf
- Mesh directory: https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/meshes
- Source revision: `5994d4faef0a9cadd3287f8de0199a67eeb2a259`.

Mesh references in the URDF were changed from `meshes/` to `../meshes/` to match this directory layout. All 64 local mesh files were verified to match the same-named files at the source revision above.

The URDF and USD have different collision geometry definitions; they should not be assumed to produce identical contacts.
