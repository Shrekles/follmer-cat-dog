# Iteration 03 report: CIFAR-only refinement of the target models

Spec: `../iter_02_purity/VALIDATION.md` §4, implemented as written. Code: `optimal_transport/tasks_iter03.py`,
`optimal_transport/modal_iter03.py`; post-processing `build_outputs.py`. All GPU work finished 15:52 PDT.

## What ran
- Smoke (14:56, 0.9 GPU-min): 20 steps at bs 256, peak 19.3 GB, so bs 256 fits an A10G with no accumulation.
- Container D (14:57–15:48, 50.4 min): dog model fine-tuned from `out/dog.pth` (EMA, fresh Adam) on CIFAR-10
  train dogs only, bs 256 bf16, 40-min cap, **3,095 steps (0.79M images, 1.29 it/s)**, lr 1e-4 → 1e-5. Then prior
  samples (seed-0 z), decode of `/vol/iter02/latents_cat_to_dog.pt`, memo check, eval.
- Container C (14:57–15:49, 50.3 min): mirror for the cat model, **2,991 steps (0.77M images, 1.25 it/s)**.
- The optional mid-snapshot prior was skipped (less than 20 min to the deadline after decoding).
- Training loss stayed flat at ≈ 0.026–0.027 throughout (it is dominated by noise; it is not informative).
- GPU: 0.9 + 50.4 + 50.3 ≈ **102 GPU-min** (in-container time; add ~1–2 min of startup per container).

New volume files (nothing overwritten): `/vol/iter03/dog_c3.pth`, `dog_c3_mid.pth`, `cat_c3.pth`,
`cat_c3_mid.pth`, `prior_dog_c3.pt`, `prior_cat_c3.pt`, `transport_cat_to_dog_iter03_c3.pt`,
`transport_dog_to_cat_iter03_c3.pt`, `evalD_*`, `evalC_*`, `memo_{dog,cat}.json` (+ `smoke_*`).

## Results (512 test inputs per direction; baseline = iteration-01 composed ODE, evaluated in iter 02 with the same code)

| | cat→dog baseline | cat→dog iter03 | dog→cat baseline | dog→cat iter03 |
|---|---|---|---|---|
| clamped cost | 101.6 ± 3.4 | **110.3 ± 3.6** | 104.6 ± 3.4 | **113.5 ± 3.5** |
| judge target rate (ceiling) | 74.8% ± 1.9 (87.7) | 76.0% ± 1.9 | 74.6% ± 1.9 (85.1) | 76.2% ± 1.9 |
| KID×1e3 vs all test target (floor 0.36 / 0.09) | 2.92 ± 0.92 | 2.61 ± 0.80 | 2.16 ± 0.64 | 2.28 ± 0.62 |
| prior samples: judge target | 79.5% | 78.3% | 80.1% | 80.7% |
| prior samples: KID×1e3 | 3.28 ± 0.77 | 2.72 ± 0.58 | 4.42 ± 0.70 | 5.20 ± 0.75 |
| memo: median NN-train dist / test median | 0.95 (old) | 0.95 (no flag) | 1.00 (old) | 1.00 (no flag) |

Prior-sample class histograms (judge argmax, 512 samples) are essentially unchanged: dog model old
`[2,0,8,57,22,407,2,11,3,0]` → new `[3,0,10,59,19,401,4,12,3,1]`; cat model old `[4,1,20,410,18,37,7,4,9,2]` → new
`[4,1,18,413,15,35,10,3,9,4]`. The leftover other-class modes were not removed.

## Success criteria (spec §4)

| criterion | cat→dog | dog→cat |
|---|---|---|
| 1. judge ≥ 79% and ≤ ceiling+2 | **no** (76.0%; +1.2 pts, < 1 se) | **no** (76.2%; +1.6 pts, < 1 se) |
| 2. KID ≤ baseline (within 1 se) | yes (2.61 vs 2.92; within noise) | yes within 1 se (2.28 vs 2.16), strictly no |
| 3. clamped cost ≤ 110 | **no**, marginal (110.3) | **no** (113.5) |
| 4. prior improves, no memorisation | **no**: judge down 1.2 pts, KID down 0.56; no memo flag | **no**: judge +0.6, KID up 0.78; no memo flag |
| 5. by eye | same as baseline (see below) | same as baseline |

**Verdict: the criteria are not met in either direction. Iteration 03 brought no measurable improvement.** All
changes in judge rate and KID are within about 1 se. The cost went up by ≈ 9 (≈ 2.5 se), which suggests the
decoders drifted slightly from the source encoders' geometry. **The best valid coupling stays the iteration-01
composed ODE: 101.6 / 104.7, judge 74.8% / 74.6% vs ceilings 87.7 / 85.1, KID 2.9 / 2.2.**

## By eye
- `cat_to_dog_compare_big.png` / `dog_to_cat_compare_big.png` (rows: input / baseline / iter03): in all 16
  columns of each, the iter03 output is nearly identical to the baseline. The same columns succeed (same-scene
  dogs in c→d columns 1, 2, 4, 6–12, 14; cats in most d→c columns), and the same ones fail: c→d column 3 (the
  yawning cat) is still unchanged, and columns 5 and 15 are still blobs. The fine-tune barely moved the decode map.
- `cat_to_dog_iter03_c3_lowest_ptarget_big.png`: the same failure modes as before. Most are unchanged cats, plus a
  red truck, a frog, a white horse and a red vehicle.
- `prior_dog_c3_lowest_ptarget_big.png`: a boat, a frog, a horse, deer, a building or vehicle scene, and cats. These
  are the same leftover CIFAR classes as `../iter_02_purity/prior_dog_lowest_ptarget_big.png`.
- `prior_same_z_cat_dog_old_new_big.png` (rows: old cat / new cat / old dog / new dog, same z): old and new are
  almost pixel-identical per column.

## Interpretation
With ~0.78M more images (~150 epochs of the 5k class images) at lr ≤ 1e-4, the EMA weights hardly move, and
the off-class modes that the all-class initialisation left behind survive. Plain continued denoising training
on the 5k class images does not remove them in this budget. Doing that would likely need a different objective
(for example, negative or contrastive training against other classes) or much longer training. Neither fits the
session.

## Problems
- The Mac disk is at ~300 MB free, and iCloud evicted freshly written figures and `gpu_log/*.json` within minutes.
  `build_outputs.py` skips figures that already exist, and it reads GPU minutes from the client logs'
  `RESULT` lines. `prior_same_z_old_new_big.png` is an early dog-only version (2 rows). The 4-row figure is
  `prior_same_z_cat_dog_old_new_big.png`.
- The first smoke launch failed with a local DNS error before any container started (0 GPU-min). The relaunch
  worked.
