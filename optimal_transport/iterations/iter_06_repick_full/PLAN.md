# Iteration 06 plan: FT pair on the full test set, and an encoder-aware re-pick of the from-scratch snapshots

Training agent, 2026-10-17. Final iteration of run 2. **No training.** Budget: run-2 total 365.0 / 600 A10G-min
before this iteration; this iteration is capped at 60 GPU-min (smoke + two containers in parallel).

Code: `optimal_transport/tasks_iter06.py` (GPU tasks), `modal_iter06.py` (deployed app `follmer-cat-dog-iter06`),
`launch_iter06.py` (spawn / fetch on the deployed app, so jobs survive a dead client). All new volume outputs go
under `/vol/iter06/` (smoke: `/vol/iter06/smoke/`); `/vol/out/`, `/vol/iter0{1..5}/`, `/vol/judge/` are only read.
Only JSON and PNG figures are downloaded (well under 10 MB); no `.pth` file is ever downloaded.

## Why

- The validator of iteration 05 recommended stopping, with two optional cheap jobs: the 1,000-image numbers for the
  best valid coupling (the fine-tuned "FT" pair) and an encoder-aware re-pick of the from-scratch snapshots.
- Iterations 04 and 05 picked snapshots by each model's own prior samples (KID / judge), which cannot see how good a
  snapshot is **as an encoder**. Same-noise distances show the dog decoders are nearly interchangeable
  (iter05 vs iter04 dog: 0.007), while the cat encoder change of iteration 05 (e9995 s31247) raised c→d cost from
  132 to 223. So the encoder choice drives cost; this iteration picks pairs on a composed-transport probe.

## Container A: FT pair on the full CIFAR-10 test set (~22 GPU-min)

- Composed probability-flow ODE `T = decode_tgt ∘ encode_src` with `/vol/out/cat.pth` and `/vol/out/dog.pth`
  (loaded with `load_score_model`, EMA weights), exactly as iteration 01: Heun, 200 steps each leg, eps = 1e-3,
  no Tweedie step, batch 256.
- New: test cats / dogs with class-indices 512-999 (488 per direction) → `/vol/iter06/transport_{dir}_FT_512_999.pt`
  (inputs, outputs, latents).
- Existing 0-511: iteration-01 files `/vol/iter01/transport_cat_to_dog_B.pt` and `/vol/iter01/transport_dog_to_cat_A.pt`
  (key `s1.0`, n = 512, the files behind 101.6 / 104.7). The task asserts their inputs equal test images 0-511 of the
  class. Determinism check: the first 16 images of each direction are re-transported with the same code and compared
  with the stored outputs (max |Δ|, cost difference).
- Report over all 1,000 per direction, and separately for 0-511 (must reproduce 101.6 / 104.7) and 512-999: cost
  unclamped and clamped to [-1, 1] (mean ± se, median), judge target / source rate on T(x) against the ceiling on all
  1,000 real test images (87.7% dog / 85.1% cat), KID ×1e3 vs all 1,000 real test images of the target class (with
  the floor: KID of 1,000 train images of the target class vs the same 1,000 test images), saturated fraction.
  Figures: input/output 16-col for images 512-543 (`_b`), lowest-p over 1,000, strict best pairs over 1,000.

## Container B: encoder-aware re-pick (~22 GPU-min)

Candidates (all saved from-scratch EMA snapshots with ≥ 12k steps, 11 per class):
- cat: iter04 `cat_s{12129,14714,17297,19874,22457}`, iter05 `cat_{e9995,e9999}_s{28347,31247,34123}`;
- dog: iter04 `dog_s{12386,14961,17541,20119,22697}`, iter05 `dog_{e9995,e9999}_s{28586,31505,34420}`.

All transports: composed PF-ODE, Heun 200 steps per leg, eps = 1e-3, no Tweedie step.

**Stage 1 (encoder screen).** cat→dog: each of the 11 cat snapshots encodes the first 64 test cats; the fixed
decoder iter04 `dog_s20119` decodes. dog→cat mirrors it (11 dog encoders, fixed decoder iter04 `cat_s17297`, first
64 test dogs). Per pair: cost unclamped / clamped (mean ± se, median), judge target and source rate on T(x),
KID ×1e3 vs all 1,000 real test images of the target (diagnostic only at n = 64), saturated fraction; per encoder:
latent Gaussianity (iteration-05 `gauss_stats`: E|z|²/d, pooled-4/8 energy ratios raw and normalised, lag-1
autocorrelation, KS). The reference pair is the iteration-04 pair (cat_s17297 → dog_s20119 for c→d, dog_s20119 →
cat_s17297 for d→c); its clamped cost on these 64 images is the cost cap `C_ref`.

**Ranking of encoders for stage 2:** encoders whose stage-1 pair has clamped cost ≤ `C_ref` come first, ordered by
judge target-rate (descending), then clamped cost (ascending); the others follow, ordered by clamped cost
(ascending). The top 3 go on.

**Stage 2 (encoder × decoder).** The latents of the top-3 encoders are decoded by every one of the 11 decoder
snapshots of the target class (same 64 images): 33 pairs per direction (3 of them are the stage-1 pairs).

**Selection rule (fixed before running).** Over every pair evaluated in stages 1 and 2 of a direction (the
iteration-04 pair included): maximise the judge target-rate on the 64 outputs subject to clamped cost ≤ `C_ref`;
ties in judge rate are broken by lower clamped cost. If the selection is the iteration-04 pair itself, the re-pick
has found nothing better and Stage 3 reproduces iteration 04.

**Stage 3 (full eval of the selected pair per direction).** First 512 test images; standard eval (`eval_runs`: cost
unclamped / clamped, judge target rate vs ceiling, KID ×1e3 vs all 1,000 test images of the target and the floor),
plus the same numbers on the held-out images 64-511 (the 64 probe images were used for selection, so their numbers
are optimistically biased), plus the memorisation check on T(x) (nearest CIFAR-train distance vs real test, the
iteration-03 code) and the snapshots' own memo ratios from iterations 04/05. FT (iteration 01) and iter04 transports
are re-scored in the same call for a side-by-side table.

Figures (4× NEAREST): input/output 16-col (`*_repick_big.png`, `*_repick_b_big.png`), lowest target-probability
outputs, strict best pairs (p_target ≥ 0.9 and p_source ≤ 0.05, clamped cost ascending), and a 32-column compare
per direction (rows input / FT / iter04 pick / new re-pick, test images 0-31). Flow and steering grids are not
redrawn: no model is trained or changed in this iteration, and the iteration-04/05 grids cover these snapshots.

## Success (validator's criteria)

- cat→dog: KID ≤ 7, clamped cost ≤ 125 and by-eye ≥ 23/32 same-scene; **or**
- dog→cat: KID ≤ 5, clamped cost ≤ 140 and by-eye ≥ 14/32.
By-eye counts on the 32-column compare (categories of the validator: (a) same scene with a convincing target animal,
(b) still the source animal, (c) broken / new scene). Validator expectation: P(success in ≥ 1 direction) ≈ 35%,
P(beats FT) ≈ 0.

## Budget

| job | estimate (A10G-min) |
|---|---|
| smoke (A and B at tiny sizes, one container) | ~4 |
| A: FT 488 + 488 and 1,000-image eval | ~22 |
| B: stages 1-3 and figures | ~22 |
| **total** | **~48 (cap 60)**; run 2 ends at ≈ 413 / 600 |

If a container would push the iteration past 60 GPU-min it is not started; if a container is still running at
30 minutes beyond its estimate it is stopped.
