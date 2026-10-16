# Iteration 01 report: judge + evaluation harness + baseline re-measurement

## What was done
- **Judge**: ResNet-18 (CIFAR variant, written from scratch, no pretrained weights), trained on the CIFAR-10
  train split only, inputs in [-1, 1], random crop (pad 4) + h-flip on GPU, SGD-Nesterov, one-cycle lr
  (max 0.2), batch 512, bf16 autocast, 30 epochs, 2.6 GPU-min. Weights: volume `judge/cifar10_judge.pth`
  (not downloaded).
- **Harness**: `optimal_transport/evaluate.py` (judge architecture, scoring, cost/rate stats, figure helper),
  `optimal_transport/tasks_iter.py` (`train_judge`, `transport_pf`, `judge_eval`; every task refuses to
  overwrite an existing path), `optimal_transport/modal_iter.py` (chains `smoke`, `A`, `B`, `eval`).
  `tasks.py`, `modal_ot.py`, `pf_ode.py`, `common/` are unchanged.
- **Baseline**: composed PF-ODE with `out/cat.pth`, `out/dog.pth`, Heun 200 steps, eps = 1e-3, full encode to
  s = 1. First 512 test cats / first 512 test dogs. ODE batches of 256.
- **Partial-encoding sweep**: encode with the source model to s* = grid point k/200 (k = 80, 120, 160 ->
  s = 0.4006, 0.6004, 0.8002), decode from s* with the target model in k Heun steps (same grid as full).
  **s* < 1 is NOT a valid coupling of mu and nu**: the decoded law is not guaranteed to be nu (the
  encoded state still carries a source-image signal, coefficient alpha(s*) below), so these costs do not
  bound W2^2. The sweep is a diagnostic of how cost trades against judged class.

## Judge (CIFAR-10 test set, 10,000 images)
| test acc | real cats called "cat" (ceiling, 1000) | real dogs called "dog" (ceiling, 1000) | mean p(cat) on cats | mean p(dog) on dogs |
|---|---|---|---|---|
| 92.95% | 85.1% | 87.7% | 0.841 | 0.871 |
Most cat errors are "dog" (70/1000) and most dog errors are "cat" (76/1000). On the first 512 (the
inputs used below): 85.4% of cats called cat, 88.5% of dogs called dog.

## Transport (n = 512 per direction; rate stderr ~0.02)
| direction | s* | alpha(s*) | valid coupling | cost E|x-T(x)|^2 | judge target-rate on T(x) | mean p(target) | still called source |
|---|---|---|---|---|---|---|---|
| cat->dog | 1.0 | 0.007 | **yes** | **101.6 +- 3.4** | **74.8%** (real-dog ceiling 87.7%) | 0.743 | 16.2% |
| cat->dog | 0.8 | 0.040 | no | 89.8 +- 3.1 | 74.0% | 0.738 | 16.4% |
| cat->dog | 0.6 | 0.161 | no | 62.2 +- 2.3 | 70.7% | 0.690 | 19.3% |
| cat->dog | 0.4 | 0.441 | no | 26.7 +- 0.8 | 56.2% | 0.553 | 31.8% |
| dog->cat | 1.0 | 0.007 | **yes** | **104.7 +- 3.4** | **74.6%** (real-cat ceiling 85.1%) | 0.727 | 12.9% |
| dog->cat | 0.8 | 0.040 | no | 92.8 +- 3.1 | 73.0% | 0.724 | 13.9% |
| dog->cat | 0.6 | 0.161 | no | 65.7 +- 2.2 | 71.9% | 0.708 | 16.4% |
| dog->cat | 0.4 | 0.441 | no | 28.4 +- 0.8 | 58.2% | 0.570 | 30.7% |

The full-encode costs reproduce iteration 00 exactly (101.6 / 104.7). Reading: with the full
(valid) coupling about 3/4 of outputs are judged the target class versus ~0.86-0.88 for real images, so
the outputs are not yet as target-like as real data; ~16% of T(cats) are still judged "cat" (matches the
visual "some barely change"). Cost keeps rising with s* (27 -> 62 -> 90 -> 102 for cat->dog), while the
judged class change mostly saturates by s* = 0.6: s* = 0.6 gets ~95% of the full target-rate at ~61-63% of the
cost (not a valid coupling, so not a W2 bound).

## Figures
`{cat_to_dog,dog_to_cat}_full_big.png` (rows input / T(x), test images 0-15), `*_full_b_big.png`
(images 16-31), `*_sweep_big.png` (rows input / s*=0.4 / 0.6 / 0.8 / 1.0). x4 NEAREST.

## GPU usage (A10G, measured inside the containers)
smoke 0.4 min; chain A (judge 2.6 + dog->cat full and s*=0.4 11.7) 14.4 min; chain B (cat->dog full + 3
partials 19.0 + dog->cat s*=0.6,0.8 11.0) 30.2 min; eval 0.1 min. **Total ~45 container-minutes** (plus
~1 min of start-up per container that is not in these numbers).

## Problems / caveats
- The judge's own ceilings are only 85-88% on cats/dogs (cat vs dog is the hardest CIFAR pair), so
  target rates should always be read relative to those ceilings. A longer schedule (e.g. 100+ epochs,
  ~9 GPU-min) would likely give ~95% overall and higher ceilings, if needed later.
- The judge is a classifier, not a distribution test: a high target rate does not prove T(x) ~ nu.
- `small.pt` is 19 MB (not ~8 MB) because `x[:64]` was saved as a view, so the full 512-input storage was
  serialised; harmless (only test images), fix with `.clone()` next time.
- Outputs are clamped to [-1, 1] before the judge; ~0.5-0.9% of output pixels lie outside that range.
- `build_outputs.py` (local post-processing) lives in this folder; smoke-test outputs are in `smoke/`.
