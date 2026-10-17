# Iteration 04 report: cat-only and dog-only models trained from scratch

**Summary.** Two DDPM++ VP U-Nets with 9.86M parameters each were trained from random init, one on the 5,000
CIFAR-train cats and one on the 5,000 dogs. Each model's own samples are purer than the fine-tuned models'
(judge 83.6% cat and 82.8% dog, against 80.1% and 79.5% before), but their KID is about 2 to 4 times worse.
The composed ODE built from them reaches a higher judge target rate (80.1% c→d, 77.3% d→c, against 74.8% and
74.6%). It also has a higher cost (132 / 140 against 102 / 105) and a 3 to 4 times worse KID (9.2 / 8.6
against 2.9 / 2.2). The same noise gives the same scene across the two independent models. It is not a
strict improvement over the current best: purity went up, while cost and sample quality went down.

## What ran (see PLAN.md and its recovery addendum)

- Training: CIFAR-10 train images of one class only, random hflip, VP continuous objective, batch 256, bf16,
  Adam 2e-4 (1,000-step warmup), grad clip 1, EMA min(0.9995, (1+t)/(10+t)), 4.3 it/s on an A10G.
- **Problem:** at about 18:15 PDT both containers were stopped by a client-side cancellation ("Received a
  cancellation signal while processing input"), after 66.9 / 65.2 training minutes (step ~17,000). This was
  not a timeout or a budget limit. Only the snapshots at 12.5/25/50% survived, and no `_last.pth` was written.
- Recovery: the surviving snapshots were evaluated (`snapeval`). The models were clearly under-trained (KID
  still falling fast, no memorisation), so each class was **resumed** from its s12k EMA snapshot with a fresh
  Adam (200-step warmup) for 40 more minutes (`chain_resume`). Total training was 87.5 minutes per class, of
  which about 19 minutes (steps 12k to 17k) were trained twice because of the cancellation. Launches now go
  through the deployed app (`modal deploy` + `launch_iter04.py spawn`), which keeps running if the local
  client dies.
- Snapshot choice rule: the lowest KID among snapshots whose memorisation ratio is at least 0.5. This picked
  **cat_s17297** and **dog_s20119**.

## Per-snapshot prior stats

256 samples from seed-0 noise each, decoded with 100 Heun steps. KID ×1e3 is measured against all 1,000 test
images of the class. The memo ratio is the median nearest-train distance of the samples divided by the same
for the first 512 real test images (a ratio below 0.5 counts as memorised).

| snapshot | train-min | cat: judge cat | cat: KID | cat: memo | snapshot | dog: judge dog | dog: KID | dog: memo |
|---|---|---|---|---|---|---|---|---|
| s3071 | 11.9 | 45.3% | 136.5 ± 7.9 | 1.05 | s3108 | 24.6% | 100.3 ± 6.9 | 1.03 |
| s6140 | 23.8 | 69.1% | 29.7 ± 3.3 | 1.11 | s6195 | 59.0% | 17.8 ± 1.8 | 1.07 |
| s12129 | 47.5 | 81.2% | 15.3 ± 1.9 | 1.11 | s12386 | 79.7% | 10.9 ± 1.4 | 1.05 |
| s14714 | 57.5 | 83.6% | 14.4 ± 1.8 | 1.07 | s14961 | 81.6% | 9.1 ± 1.3 | 1.00 |
| **s17297** | 67.5 | **83.6%** | **14.1 ± 1.7** | 1.06 | s17541 | 81.6% | 8.7 ± 1.4 | 0.99 |
| s19874 | 77.5 | 83.6% | 14.2 ± 1.8 | 1.05 | **s20119** | **82.8%** | **8.2 ± 1.3** | 0.99 |
| s22457 | 87.5 | 87.1% | 14.5 ± 1.9 | 1.02 | s22697 | 84.0% | 8.9 ± 1.3 | 0.96 |

Reference points:
- Judge ceilings: cat 85.1%, dog 87.7%.
- KID floor (256 train images vs test): about 0 (−0.16 cat, 0.19 dog).
- Fine-tuned models' priors (iteration 02, n = 512): 80.1% cat / KID 4.4, and 79.5% dog / KID 3.3.

KID has levelled off since about 60 minutes, while the judge rate keeps creeping up. The memorisation ratio
falls slowly but stays far from 0.5, and no sample lies within 10% of the test median distance to its nearest
training image. Picking by KID alone took a mid snapshot. The final cat snapshot (s22457) is the purest, with
87.1%, but its KID is not significantly different.

## Composed ODE (valid coupling, s* = 1, w = 0), first 512 test cats and dogs, 200 Heun steps

| direction | model | cost | cost clamped | judge target (ceiling) | KID ×1e3 (floor) | source class on outputs |
|---|---|---|---|---|---|---|
| cat → dog | **from scratch (iter 04)** | 132.4 ± 3.1 | 132.0 ± 3.1 | **80.1% ± 1.8** (87.7%) | 9.2 ± 1.3 (0.36) | 12.9% |
| cat → dog | fine-tuned (current best) | 101.6 ± 3.4 | 101.6 ± 3.4 | 74.8% ± 1.9 | 2.9 ± 0.9 | 16.2% |
| dog → cat | **from scratch (iter 04)** | 141.3 ± 3.2 | 139.9 ± 3.1 | **77.3% ± 1.9** (85.1%) | 8.6 ± 1.1 (0.09) | 12.9% |
| dog → cat | fine-tuned (current best) | 104.7 ± 3.4 | 104.6 ± 3.4 | 74.6% ± 1.9 | 2.2 ± 0.6 | 12.9% |

Outputs that are neither cat nor dog drop from 46 to 36 of 512 (c→d) and from 64 to 50 (d→c). That is
consistent with models that never learned the other classes, but they do not reach zero. Pixels outside
[−1, 1]: 1.8% and 3.7% (against 0.5% and 0.9% before), so the small nets overshoot more.

## Steering-strength grid (cat → dog, statistics over the first 64 test cats; `steering_grid.json`)

| setting | s*=0.2 | 0.4 | 0.6 | 0.8 | **1.0 (valid)** | w=0.25 | w=0.5 | w=1.0 |
|---|---|---|---|---|---|---|---|---|
| mean cost (clamped) | 9.7 | 40.2 | 75.5 | 98.6 | **117.2** | 172.8 | 236.1 | 382.7 |
| judge dog-rate | 43.8% | 68.8% | 82.8% | 82.8% | **81.2%** | 89.1% | 95.3% | 98.4% |

Only s* = 1, w = 0 is a coupling. The other columns are diagnostics. The dog-rate saturates by s* = 0.6, the
same pattern as with the fine-tuned models. Guidance raises the dog-rate only by redrawing the scene, at 1.5
to 3 times the cost.

## Same noise, two independently trained models

`same_noise_both_models_big.png` decodes 16 seed-0 noises with the new cat and dog models, plus the old
fine-tuned models. Over 256 shared noises, the mean |cat(z) − dog(z)|² is **116 for matched noise against
1,621 for mismatched noise** (ratio 0.071). The fine-tuned pair gives 89 against 1,526 (ratio 0.058), and the
new and old models of the same class give ratios of about 0.05. So **yes: two models trained from scratch,
independently, on disjoint image sets map the same noise to the same scene** (layout, background, colours
and pose), with a clearly different animal. The match is only slightly looser than for the two fine-tuned
models, which share an initialisation. Both models of each matched pair are judged the right class in 68% of
pairs (64% for the fine-tuned pair).

## By eye

- `cat_to_dog_scratch_big.png`: about 10 of the first 16 columns are convincing same-scene dogs. A few change
  the scene (the cat by the window becomes a large dog face) or stay blurry or cat-like.
- `cat_to_dog_scratch_lowest_ptarget_big.png`: the failures are mostly cats left unchanged, plus one green
  frog-like input that stays as it is.
- `best_pairs_big.png` (307 of 512 qualify: P(dog|out) ≥ 0.9 and P(cat|in) ≥ 0.5): cost 19 to 51. Several of
  the lowest-cost "dogs" still look partly cat-like; the judge over-credits them.
- `flow_grid_big.png`: with trained models the raw PF-ODE states at s ≥ 0.2 already look like noise, so this
  figure shows little. Next time, show the Tweedie denoised estimate x̂₀(x_s) next to each state.

## Figures (all 4× NEAREST)

`flow_grid_big.png`, `flow_grid_dog_to_cat_big.png`, `steering_grid_big.png`, `best_pairs_big.png`,
`best_pairs_dog_to_cat_big.png`, `same_noise_both_models_big.png`, `{cat_to_dog,dog_to_cat}_scratch{,_b}_big.png`,
`{cat_to_dog,dog_to_cat}_scratch_lowest_ptarget_big.png`, `prior_{cat,dog}_s*_big.png` and
`prior_*_lowest_ptarget_big.png` (one per snapshot).

## New models (Modal volume `follmer-cat-dog`)

- Chosen: `iter04/cat_s17297.pth` and `iter04/dog_s20119.pth`.
- All EMA snapshots: `iter04/{cat,dog}_s*.pth`.
- Full training state (resumable): `iter04/{cat,dog}_last.pth` (step 22,457 / 22,697).
- Latents: `iter04/latents_{cat_to_dog,dog_to_cat}.pt`. Transports: `iter04/transport_*_scratch.pt`.
- All checkpoints store `config_overrides`. Load them with `common/score_sde_backend.load_score_model_auto`.

## GPU

234.1 A10G-min for this iteration (budget 260), including the 2 × 68.5 min of the cancelled runs; run-2 total
234.1 / 600. See `../BUDGET_RUN2.md`.

## Suggested next steps

1. The from-scratch models fix part of the purity problem but cost sample quality (KID 8 to 14 against 3 to
   4). Capacity is the likely limit: KID has levelled off while the memorisation ratio stays near 1. A wider
   net (nf = 96 or 128, num_res_blocks = 2) trained for longer is the natural next test. Watch the memo ratio.
2. Cheap alternative: resume the current nets from `_last.pth` with lr decay (2e-4 to 2e-5) for 30 to 40
   minutes. A decaying lr often lowers KID at fixed capacity.
3. Because independent models share the same scene for the same noise, other model pairs can be swapped into
   the composed ODE freely. For example, try the fine-tuned cat encoder with the from-scratch dog decoder.
   That would need about 2 minutes of decode GPU time using the saved iter-02 latents.
