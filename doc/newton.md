# Newton Upgrade Guide (rough-terrain NaN fix)

This project is based on **IsaacLab 3.0 GA** (the `v3.0.0` tag). The default stack
(newton 1.6.1) runs fine, but **long training runs on rough terrain require upgrading
newton to 1.7.0**. The why, the how, and which IsaacLab checkout to use are below.

## 1. Why the upgrade is required: the old contact kernel NaNs on long rough-terrain runs

IsaacLab 3.0 GA pins `newton[sim]==1.6.1` / `mujoco-warp~=3.12` — the old contact
kernel. Measured behavior (newton 1.5.0 / mujoco-warp 3.11, same kernel generation):

- **Rough (generator) terrain**: height-field / mesh terrain produces non-finite qacc
  at a rate of roughly 10⁻⁴–10⁻⁵ per env-step (especially the single-cell GJK
  approximation in the height-field path); rsl_rl's `check_for_nan` then aborts
  training outright. More iterations and more environments make the crash a certainty.
- **Long runs on plane terrain**: the same low-frequency NaN exists — G1-Warp crashed
  once every ~13k iters (≈1.3×10⁹ env-steps), making a clean 30k run practically
  impossible.

After upgrading to newton 1.7.0 (which brings mujoco / mujoco-warp 3.15.0, the new
contact kernel):

- G1-Warp completed 30k iters with zero NaN (the old stack crashed at 12k / 13.5k;
  the new stack's survival is ≈ 2.3× the old MTTF).

## 2. How to upgrade newton

After installing IsaacLab, run (one step — newton 1.7.0 automatically brings
mujoco==3.15.0 and mujoco-warp==3.15.0):

```bash
pip install "newton[sim]==1.7.0" "warp-lang==1.17.0"
```

Two gotchas you must know:

1. **`warp-lang` must be pinned to 1.17.0.** warp 1.18 bundles the CUDA 13.4 runtime;
   on GPU drivers below 580 (CUDA 12.x) the GPU becomes invisible (`cuda:0`
   disappears — it errors out at init). warp 1.17 bundles CUDA 12.9, which works on
   12.x drivers. newton 1.7.0 requires `warp>=1.17`, so the two are compatible.
2. **The torch CUDA build must match the driver.** isaacsim pulls torch
   `2.11.0+cu130` by default; on drivers below 580 the GPU is likewise invisible —
   switch to the cu128 build:

   ```bash
   # NOTE: the version spec torch==2.11.0 matches BOTH +cu130 and +cu128, so a plain
   # install is a no-op — uninstall first, then install
   pip uninstall -y torch torchvision
   pip install torch==2.11.0 torchvision==0.26.0 \
       --index-url https://download.pytorch.org/whl/cu128
   ```

   The cu128 build works on both 570-series (CUDA 12.8) and 580-series (CUDA 13.x)
   drivers.

Verify:

```bash
python -c "
import newton, warp as wp, torch
print('newton', newton.__version__, '| warp', wp.__version__, '| torch', torch.__version__)
print('cuda avail:', torch.cuda.is_available())
"
```

For faster downloads in China, use the Huawei mirror (for the first command):

```bash
pip install -i https://mirrors.huaweicloud.com/repository/pypi/simple \
    "newton[sim]==1.7.0" "warp-lang==1.17.0"
```

## 3. Which IsaacLab version to check out

```bash
cd IsaacLab
git checkout v3.0.0        # the 3.0 GA tag, reproducible
```

- Use the **tag**, not a branch head: a tag is a frozen release snapshot; the
  `release/3.0.0` branch = tag + backported fixes. Upgrade wholesale when a patch
  tag (e.g. `v3.0.1`) ships — do not track the branch head.
- This repo's `scripts/rsl_rl/train.py` / `play.py` are written against the 3.0 GA
  `resolve_task_config()` API (GA removed the old `hydra_task_config`), so IsaacLab
  must be 3.0 GA or newer; older checkouts (3.0 EA / 2.x) are incompatible.

### The three adaptations needed for the v3.0.0 tag + newton 1.7.0

The `v3.0.0` tag's isaaclab_newton is written against newton 1.6.1's API, and
newton 1.7.0 renamed several internal attributes. Without these fixes, env cloning
fails immediately with
`AttributeError: 'ModelBuilder' object has no attribute '_cloth_label'`.

The adaptation ships as a patch file in this repo:
[`doc/newton17-compat.patch`](newton17-compat.patch)
(taken from upstream develop
[#8381](https://github.com/isaac-sim/IsaacLab/pull/8381)
"Update Newton and preserve runtime compatibility"). The patch only touches the
IsaacLab repository. From the IsaacLab checkout root:

```bash
cd IsaacLab
git checkout v3.0.0
git apply /path/to/basic_loco_lib/doc/newton17-compat.patch   # revert with git apply -R on the same file
```

The three files the patch touches (all cloth/soft-body bookkeeping; for rigid robots
these renames are equivalent to renaming empty lists — no effect on training
results):

| IsaacLab file | Change |
|---|---|
| `source/isaaclab_newton/isaaclab_newton/cloner/replicate.py` | `builder._cloth_label` / `_soft_label` (and `_particle_start`) → `surface_label` / `volume_label` (and `_surface/_volume_particle_start`) |
| `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py` | the same set of renames (incl. `_particle_end`) |
| `source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor.py` | `contact_view.sensing_obj_transforms` → `sensing_transforms` (debug-visualization path only) |

The rest of #8381 (actuator checkpoints, cables, hard-reset lifecycle) is not on
this project's code path; the patch does not include it and does not need to. Once
IsaacLab ships a patch tag containing the fix (≥ v3.0.1), bump the tag and remove
this patch with `git apply -R`.

> **Why not just check out a branch that contains the fix?** (verified 2026-10) The
> fix only landed on the `develop` / `main` development lines; no official tag
> includes it — `v3.0.0` is still the newest tag. `main` is hundreds of unreleased
> commits ahead of v3.0.0, and on develop these three files carry changes from other
> commits too, so syncing whole files would drag in extra dependencies. Tag + this
> patch is currently the only reproducible approach.

## Checklist

- [ ] IsaacLab: `git checkout v3.0.0` then `git apply /path/to/basic_loco_lib/doc/newton17-compat.patch`
- [ ] `pip install "newton[sim]==1.7.0" "warp-lang==1.17.0"`
- [ ] Driver < 580: switch torch to the cu128 build (uninstall first)
- [ ] The verification script prints `newton 1.7.0 | warp 1.17.0 | torch 2.11.0+cu128`, `cuda avail: True`
- [ ] First headless run in a fresh env: add `OMNI_KIT_ACCEPT_EULA=YES` (or run once interactively to accept)

## Note: retirement of this repo's old stable-newton patch

`tasks/amp/physics/newton_cholesky.py` (a port of mujoco_warp
[#1488](https://github.com/google-deepmind/mujoco_warp/pull/1488): pivot floor +
Hessian-factor reuse) is compatible only with `mujoco-warp==3.11.0` (the EA stack)
and raises `RuntimeError` on the GA stack (3.12+). The AMP warp task now uses the
default `NewtonMJWarpManager` again: numerical stability is provided by the newton
1.7.0 stack (see the evidence in section 1), so the pivot floor is no longer needed.
The patch file is archived rather than deleted — if NaN ever recurs under more
extreme stiff-contact scenarios, its approach can be re-ported against mujoco-warp
3.15's internal APIs (the upstream PR is still unmerged in any release as of
2026-10).
