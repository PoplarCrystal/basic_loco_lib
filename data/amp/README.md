# AMP motion data provenance

The maintainer identifies `WalkandRun/` as derived from BONES-SEED and
`Recovery/` as derived from Ubisoft LAFAN1. Original dataset revisions and the
exact retargeting/conversion commands still need to be recorded.

## Sources and terms

| Directory | Original source | License reference |
|---|---|---|
| `WalkandRun/` | [BONES-SEED by Bones Studio](https://bones.studio/datasets/seed), [official download](https://huggingface.co/datasets/bones-studio/seed) | [BONES-SEED Dataset License Agreement](https://bones.studio/info/seed-license) |
| `Recovery/` | [Ubisoft La Forge Animation Dataset (LAFAN1)](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | [CC BY-NC-ND 4.0, upstream license](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt) |

References checked on 2026-10-23. BONES-SEED has its own access eligibility and
attribution terms, and prohibits public dataset mirrors and redistribution.
The Apache license of its viewer code does not license the motion data.
LAFAN1 is published with noncommercial and no-derivatives terms. Redistribution
permission for these converted robot trajectories has not been established.
The repository's Apache-2.0 code license does not apply to these recordings.

## Before public release

Do not publish these NPZ files without confirming the applicable permissions.
Acknowledgments alone do not establish redistribution rights. Obtain any necessary
permission or distribute code and acquisition/conversion instructions separately.
Because the files already occur in Git history, deleting them from the latest
revision alone would not remove them from a public clone of that history.
No data files or Git history have been removed by this documentation change.

Download original data directly from the official sources under their terms.
For reproduction, the maintainer still needs to document the exact source revision,
source clip paths, retargeting tool/version, input frame rate and conversion command.
The repository provides `scripts/mimic/csv_to_npz.py` for compatible G1 CSV input,
but its use for these existing clips has not been verified.

## Credits

Locomotion motion capture: Bones Studio, BONES-SEED.
Recovery motion capture: Ubisoft La Forge, LAFAN1.

LAFAN1 reference: Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai,
and Christopher Pal, *Robust Motion In-Betweening*, ACM Transactions on Graphics
(SIGGRAPH), 39(4), 2020.

## File inventory

SHA-256 hashes identify the exact files covered by the eventual source records.

| File | SHA-256 |
|---|---|
| `Recovery/fallAndGetUp1_subject1.npz` | `2073b24d18cdb19908a5d6acae4e20312dd54ed8d5f693c3c0fc286e2200a720` |
| `WalkandRun/arc_jog_left_loop_002__A029.npz` | `a52f55ae5f6c466a379e0611b7d4bdfc95d097f650344921ba45c8034baf3780` |
| `WalkandRun/arc_walk_left_loop_001__A029.npz` | `394b744b2fcd051d1fb4066a5ccc23790465fb37c32773277ea7f3bd88270e9e` |
| `WalkandRun/idle_turn_270_001__A048.npz` | `421138293e13dab6175d00e86561cd02b3d0332dd677b5ac8c8eece2ca67b097` |
| `WalkandRun/idle_turn_360_001__A047.npz` | `5abe406cf093fce62ef76b9326f1b6604f292a59ec55085864230e89fcf11e04` |
| `WalkandRun/jog_arc_cw_loop_004__A045.npz` | `3924c48a10757c1a412fad1dfc18aba31193eaeb0637c7746150a7eadd5d147a` |
| `WalkandRun/jog_backward_loop_002__A022.npz` | `65d50795be6d5728dce155d9eab380492cad9f18ae8a9d934a7078035e7b8805` |
| `WalkandRun/jog_backward_loop_003__A023.npz` | `b9761af0bd904aab4faf3dc45c6ac21965180aef1879a190acb9f651a2f08970` |
| `WalkandRun/jog_forward_loop_003__A021.npz` | `19195f45e592fbcf6d1da242c13f4e5ac70d9d131914c35bd3a4606068474c73` |
| `WalkandRun/jog_forward_loop_003__A022.npz` | `5ef5bf69dab2d1044acafcae13df980733a74918fe9209a11e2c6db59a998d92` |
| `WalkandRun/step_rotate_idle_000_002__A026.npz` | `012a1aa66798b34e20fc1d4e7e78a8e18d052fc50ca8a2d844768a1cd203ff26` |
| `WalkandRun/walk_arc_cw_loop_002__A046.npz` | `e42388fb8019fab27b5807b8cd530f7edf59164aaf8a8ae9367efacf16e6cf56` |
| `WalkandRun/walk_backward_loop_001__A022.npz` | `c2e51c855725ecff363519de5764b6ec2c33bed35e2acf2070ea73b18393e14f` |
| `WalkandRun/walk_backward_loop_001__A024.npz` | `3ae5059e5a79cd9333d308c85e1f6f2119b75c4395409bdae92cc35db1d26e98` |
| `WalkandRun/walk_forward_loop_002__A022.npz` | `887b63fa373da470883f7fe00e6ce90a135e24863c74d552c75d8a4c0d82dd2e` |
| `WalkandRun/walk_forward_loop_002__A024.npz` | `1049c9261f515d18113ac7c5c47575c5062d7a80f2ada3722419a94472be93fb` |
| `WalkandRun/walk_sideway_left_loop_002__A021.npz` | `b3a7b70c70d99da9bfcf815bcf55d728c0e6a68dc616b656f4282f82b310488c` |
| `WalkandRun/walk_sideway_right_loop_001__A022.npz` | `0a2c9a877460d67be6781ae9eb1f7ca0f2df7888207884e24a5860d16c71cd2d` |
