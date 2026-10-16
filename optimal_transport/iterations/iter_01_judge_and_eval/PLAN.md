# Iteration 01 plan: judge + standard evaluation harness + baseline re-measurement

Written 12:45 PDT, deadline 13:45 PDT (all GPU jobs finished, results written).

## What
1. **Judge.** ResNet-18 (CIFAR variant: 3x3 stem, no max-pool), written from scratch, trained on the
   CIFAR-10 TRAIN split only (`/vol/data/cifar10_train.pt`), inputs in [-1, 1] (per-channel
   normalisation is inside the network). GPU-side augmentation (random crop pad 4, horizontal flip),
   SGD-Nesterov + OneCycle, bf16 autocast, ~30 epochs, hard time cap 18 min. No label smoothing (we
   want honest probabilities). Weights -> `/vol/judge/cifar10_judge.pth` (never downloaded).
   Reported: test accuracy, real test cat->"cat" and dog->"dog" rates (calibration ceilings).
2. **Harness.** `optimal_transport/evaluate.py` (plain functions: judge architecture, judge scoring,
   cost stats, figure helpers), `optimal_transport/tasks_iter.py` (GPU tasks: `train_judge`,
   `transport_pf`, `judge_eval`), `optimal_transport/modal_iter.py` (Modal entry, `--smoke`).
   Existing `tasks.py` / `modal_ot.py` untouched.
3. **Baseline.** Composed PF-ODE, `out/cat.pth` / `out/dog.pth`, Heun 200 steps, eps=1e-3, full
   encode to s=1, on the first 512 test cats and first 512 test dogs, both directions.
4. **Partial-encoding sweep (diagnostic).** Encode with the source model to s* in {0.4, 0.6, 0.8}
   (same step grid as the full encode: s* = grid points k = 80, 120, 160 of 200, i.e.
   s = eps + k(1-eps)/200), decode from s* with the target model using k Heun steps. **s* < 1 is NOT a
   valid coupling of mu and nu** (the decoded law need not be nu); it only shows how cost trades
   against judged "dogness".

## Scheduling (max 2 A10G containers)
Pass = one 200-step Heun ODE over 512 images ~ 5 min (from the iter-00 log: 3 passes in 893 s).
- Container A: train_judge (~15 min) then dog->cat full encode (saving s=0.4 state) + full decode +
  s*=0.4 decode (~12 min).
- Container B: cat->dog full encode (saving 0.4/0.6/0.8 states) + full decode + 3 partial decodes
  (~19 min), then dog->cat encode to 0.8 (saving 0.6) + s*=0.6, 0.8 decodes (~11 min).
- Then container C: judge_eval (~3 min) -> stats JSON + small tensor file on `/vol/iter01/`,
  downloaded here; figures built locally.
Smoke test of all three tasks first (tiny sizes, `smoke_` file names).
Estimated GPU: ~65-70 min including smoke.
