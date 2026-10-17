# Iteration 05 report: cosine-lr resume, mixed encoder/decoder pairs, Tweedie step

**Summary.** The cosine lr decay with a long EMA did **not** help. Prior KID rose steadily with further training
in both classes:
- cat: 17.3 → 20.4 at n = 512, against 13.7 for the iteration-04 pick at the same protocol;
- dog: 9.8 → 11.7, against 8.75.

The memo ratio of the 0.9995 EMA fell to 0.85-0.89. The small nets look past their best on 5,000 images rather than
limited by lr noise.

The composed ODE with the iteration-05 picks is worse than iteration 04 for cat→dog: cost 223 against 132, KID 16.9
against 9.2. For dog→cat it has a better KID (6.0 against 8.6) but a higher cost (160 against 140).

The mixed encoder/decoder pairs fail clearly (KID 27-48). The reason is a latent-scale mismatch between model
families, which the Gaussianity check makes visible:
- FT latents decoded by scratch decoders come out washed out;
- scratch latents decoded by FT decoders come out over-saturated.

The final Tweedie step makes KID slightly worse in all 10 transport sets.

**No criterion is met, and the fine-tuned composed ODE stays the best valid coupling.**

## What ran

| job | GPU-min | notes |
|---|---|---|
| smoke | 2.2 | 20 resume steps, snap eval n = 16 / 5 steps, eval05 n = 8 / 10 steps; every figure and code path ran |
| A: train_cat | 54.2 | 45-min cosine resume from `iter04/cat_last.pth`: 11,666 steps at 4.32 it/s. Adam state loaded (Adam step 10,328). Then snapshot eval (6 snapshots, 8.5 min) |
| B: train_dog | 54.0 | the same for dog: 11,723 steps, Adam state loaded (step 10,311) |
| eval | 20.5 | composed ODE 512 + 512 ± Tweedie; mixed pairs M1-M4; FT and iter04 re-scored ± Tweedie; Gaussianity; all figures |
| **iteration total** | **130.9** | run-2 total **365.0 / 600** (235.0 left) |

A CPU-only helper (`costq_iter05.py`, no GPU) computed per-image cost quantiles from the saved transports
(`cost_quantiles.json`).

**Deviation and problem.** After seeing the snapshot stats I added two mixed pairs, M5 and M6 (FT latents → iter-04
decoders), and redeployed. The eval spawn still ran the previous deployment, so **M5 and M6 did not run**. I did not
rerun them, and the code was reverted to what ran. On the same noise, the iter-05 and iter-04 decoders produce nearly
the same images: matched / mismatched cost 0.007 for dog and 0.032 for cat. So M5 and M6 would almost certainly
repeat M1 and M2, and rerunning would have cost ~8 GPU-min for no new information.

## 1. Per-snapshot prior stats (n = 512, prior_z seed 0, 200 Heun steps, eps 1e-3; KID ×1e3 vs all 1,000 test images)

Ceilings: cat 85.1%, dog 87.7%. KID floor (512 train vs test): ≈ 0.

| class | snapshot | EMA | train-min | judge | KID ± se | KID + Tweedie | memo ratio | saturated |
|---|---|---|---|---|---|---|---|---|
| cat | **iter04 ref s17297** | 0.9995 | 67.5 | 83.6% | **13.7 ± 1.2** | 11.6 | 1.02 | 2.9% |
| cat | e9999 s28347 (50%) | 0.9999 | 110.0 | 82.8% | 17.3 ± 1.5 | 14.1 | 1.02 | 1.8% |
| cat | **e9995 s31247 (75%) = pick** | 0.9995 | 121.3 | 84.2% | 18.4 ± 1.3 | 16.2 | 0.90 | 1.0% |
| cat | e9999 s31247 | 0.9999 | 121.3 | 80.7% | 19.2 ± 1.5 | 15.5 | 1.02 | 1.6% |
| cat | e9995 s34123 (100%) | 0.9995 | 132.5 | 83.4% | 20.1 ± 1.3 | 17.9 | 0.89 | 0.6% |
| cat | e9999 s34123 | 0.9999 | 132.5 | 80.5% | 20.4 ± 1.5 | 16.4 | 1.00 | 1.6% |
| dog | **iter04 ref s20119** | 0.9995 | 77.5 | 81.8% | **8.75 ± 0.88** | 9.5 | 0.96 | 1.8% |
| dog | **e9999 s28586 (50%) = pick** | 0.9999 | 110.0 | 80.3% | 9.8 ± 0.9 | 10.1 | 0.97 | 1.2% |
| dog | e9995 s31505 (75%) | 0.9995 | 121.3 | 79.3% | 10.8 ± 0.9 | 12.5 | 0.87 | 0.8% |
| dog | e9999 s31505 | 0.9999 | 121.3 | 78.7% | 10.8 ± 1.0 | 10.5 | 0.97 | 1.2% |
| dog | e9995 s34420 (100%) | 0.9995 | 132.5 | 78.7% | 11.9 ± 1.0 | 13.7 | 0.85 | 0.6% |
| dog | e9999 s34420 | 0.9999 | 132.5 | 78.7% | 11.7 ± 1.0 | 11.2 | 0.96 | 1.2% |

- **Pick rule** (the iter-04 reference is not eligible):
  - cat: lowest KID e9999 s28347 (17.3 ± 1.5). Within 1 se is e9995 s31247 (18.4), which has the higher judge rate
    (84.2% against 82.8%), so it was picked.
  - dog: nothing else lies within 1 se of e9999 s28586.
  - Had the reference been eligible, it would have won in both classes.
- **KID gets worse monotonically with training** in both classes and for both EMAs. The training loss fell slightly
  (0.029 → 0.027) as the lr decayed. The 0.9995 EMA's memo ratio dropped from 1.02 / 0.96 to 0.89 / 0.85, while the
  0.9999 EMA stays near 1. This is the start of over-fitting, not under-fitting. By eye, the decayed samples are
  smoother and lower in texture than the iteration-04 samples (`prior_cat_e9995_s31247_big.png` against
  `prior_cat_ref04_s17297_big.png`).
- **Saturated samples** (more than 50% of pixels clipped) occur in 0.6-2.9% of samples, against the ≤ 0.5% target. The
  iteration-04 reference is the worst on this (2.9% cat). Decay lowers the rate, but not to the target.
- **The Tweedie step** helps the cat priors' KID (by 2-4 points) but not the dog priors (it is mixed). On the transports it
  hurts (see below).

## 2. Composed ODE and mixed pairs (first 512 test cats / dogs, 200 Heun steps; valid couplings s* = 1, w = 0)

KID floors: dog 0.36, cat 0.09. Judge ceilings: dog 87.7%, cat 85.1%. Costs are mean |x − T(x)|² ± se, with outputs
clamped to [−1, 1].

| direction | coupling | cost | cost clamped | median (clamped) | judge target | source on out | KID ×1e3 | KID + Tweedie | saturated / >10% clipped |
|---|---|---|---|---|---|---|---|---|---|
| c→d | **FT (iter01, best)** | 101.6 ± 3.4 | 101.6 | 76.4 | 74.8% | 16.2% | **2.92 ± 0.92** | 4.02 | 0 / 6 |
| c→d | iter04 scratch | 132.4 ± 3.1 | 132.0 | 115.7 | **80.1%** | 12.9% | 9.22 ± 1.35 | 10.44 | 2 / 22 |
| c→d | **iter05** | 230.2 ± 4.9 | 222.7 | 208.6 | 74.8% | 18.0% | 16.85 ± 1.68 | 17.97 | 10 / 119 |
| c→d | M1: FT cat enc → iter05 dog dec | 188.4 ± 4.3 | 188.1 | 168.9 | 82.8% | 14.1% | 33.3 ± 2.6 | 37.3 | 2 / 8 |
| c→d | M3: iter05 cat enc → FT dog dec | 398.3 ± 7.1 | 389.8 | 370.8 | 47.5% | 26.6% | 48.4 ± 2.8 | 48.7 | 4 / 180 |
| d→c | **FT (iter01, best)** | 104.7 ± 3.4 | 104.6 | 85.5 | 74.6% | 12.9% | **2.16 ± 0.64** | 3.06 | 0 / 10 |
| d→c | iter04 scratch | 141.3 ± 3.2 | 139.9 | 126.8 | 77.3% | 12.9% | 8.62 ± 1.06 | 10.23 | 3 / 62 |
| d→c | **iter05** | 160.7 ± 3.4 | 160.4 | 148.8 | 76.4% | 13.3% | **5.98 ± 0.87** | 7.37 | 0 / 13 |
| d→c | M2: FT dog enc → iter05 cat dec | 149.0 ± 3.8 | 148.9 | 130.2 | 87.3% | 7.4% | 27.3 ± 2.0 | 31.2 | 0 / 9 |
| d→c | M4: iter05 dog enc → FT cat dec | 287.4 ± 5.5 | 284.8 | 266.9 | 47.7% | 17.6% | 33.2 ± 2.0 | 33.3 | 1 / 83 |

- **With and without the Tweedie step:**
  - Cost and judge rate barely change (≤ 0.6 cost, ≤ 1.4 judge points).
  - KID gets worse in all 10 output sets, by 0.0 to +4.0 (FT +1.1 / +0.9, iter04 +1.2 / +1.6, iter05 +1.1 / +1.4).
    Each difference is about 1 se, but the sign is the same every time.
  - A plausible reading: removing the residual σ ≈ 0.01 noise makes the outputs smoother than real CIFAR images in
    Inception feature space.
  - **Not adopted.** Plain outputs stay the reported numbers.
- **iter05 cat→dog is much worse than iter04.** The cost increase is broad, not driven by outliers: the median rises
  from 116 to 209, and the cost excluding saturated outputs is 221. 119/512 outputs have more than 10% of pixels
  clipped, against 22 for iter04. The outputs are higher in contrast. The iteration-05 cat encoder (e9995 s31247)
  produces the least Gaussian latents of all sets (§3). The dog decoder turns that excess low-frequency energy into
  over-contrasted images.
- **iter05 dog→cat:** KID 6.0, the best for any scratch pair and the only value at or below the spec's 6, but the cost is 160. The
  judge rate is 76.4%, just under the 77% that the spec requires.
- **Mixed pairs fail:**
  - FT latents → scratch decoders (M1, M2) give low-contrast, washed-out images with lost backgrounds. The judge still
    gives them high target rates (82.8%, and 87.3%, which is above the cat ceiling), but KID is 27-33.
  - Scratch latents → FT decoders (M3, M4) give posterised, over-saturated images (KID 33-48) and very high cost.
  - The cause is that each model family has its own latent statistics (§3). A decoder only works well on latents
    with its own encoder's energy profile. Same noise → same scene still holds for N(0, I) noise across all four
    models, but encoded real images are not N(0, I).

## 3. Gaussianity of the latent sets (n = 512 each)

The pooled ratio is k²·E[avgpool_k(z)²], which is 1 for iid N(0, 1). "norm." divides it by the set's own E[z²]. A set
is flagged (spec rule) if a raw pooled ratio differs from fresh N(0, I) by more than 3 × its bootstrap se.

| latent set | std | E‖z‖²/d (q01-q99) | pool4 raw ± se | pool8 raw ± se | pool4 / pool8 norm. | lag-1 h / v | KS vs N(0,1) | flagged |
|---|---|---|---|---|---|---|---|---|
| N(0, I) fresh, seed 1 | 1.000 | 1.000 (0.94-1.06) | 0.998 ± 0.005 | 0.997 ± 0.009 | 1.00 / 1.00 | 0.000 / −0.001 | 0.0004 | — |
| N(0, I) prior_z, seed 0 | 1.000 | 0.999 (0.95-1.05) | 0.995 ± 0.004 | 0.992 ± 0.009 | 1.00 / 0.99 | 0.000 / 0.000 | 0.0008 | no |
| FT cat encoder (iter02) | 0.881 | **0.777** (0.43-0.95) | 0.993 ± 0.006 | 1.009 ± 0.010 | **1.28 / 1.30** | 0.115 / 0.142 | 0.037 | no* |
| FT dog encoder (iter02) | 0.891 | **0.794** (0.63-0.96) | 1.023 ± 0.006 | 1.026 ± 0.010 | 1.29 / 1.29 | 0.122 / 0.147 | 0.034 | yes (4.4 se) |
| iter04 cat encoder | 0.968 | 0.938 (0.77-1.01) | 1.180 ± 0.008 | 1.172 ± 0.012 | 1.26 / 1.25 | 0.114 / 0.136 | 0.017 | yes (23 se) |
| iter04 dog encoder | 0.965 | 0.932 (0.86-1.01) | 1.196 ± 0.007 | 1.199 ± 0.011 | 1.28 / 1.29 | 0.120 / 0.142 | 0.017 | yes (27 se) |
| iter05 cat encoder | 0.948 | 0.899 (0.60-1.03) | **1.320 ± 0.010** | **1.404 ± 0.017** | **1.47 / 1.56** | 0.149 / 0.165 | 0.023 | yes (32 se) |
| iter05 dog encoder | 0.956 | 0.914 (0.83-1.01) | 1.228 ± 0.008 | 1.257 ± 0.013 | 1.34 / 1.38 | 0.132 / 0.150 | 0.019 | yes (29 se) |

- **No encoder produces N(0, I) latents.** Every set carries 25-56% excess low-frequency energy relative to its own
  variance, plus lag-1 spatial autocorrelation of 0.11-0.17 (0 for N(0, I)). This is image structure that survives
  the encoder (the window bars in the flow grid).
- *The FT cat set escapes the spec's flag only because its total energy is low (E‖z‖²/d = 0.78, with a wide spread down
  to 0.43). That cancels the excess low-frequency energy in the raw ratio. Its normalised ratio, 1.28, is as
  non-Gaussian as the scratch sets.
- **The two families differ in latent scale.** FT encoders shrink the latents (energy 0.78-0.79). Scratch encoders
  keep the energy near 0.9-0.94 but with more low-frequency excess. That matches the mixed-pair failures: lower-energy
  FT latents come out washed out, higher-energy scratch latents come out over-saturated. Since the latents are not
  N(0, I), the output law of a mixed pair is not the decoder's prior law, and KID confirms it is far from the data.
- The iteration-05 cat encoder is the least Gaussian set, which fits its cat→dog cost increase.

## 4. Required figures (4× NEAREST)

- `flow_grid_big.png`, `flow_grid_dog_to_cat_big.png`: these now show **denoised estimates** x̂₀(x_s) (odd rows) and
  x_s/α_s (even rows) at s = 0.1, 0.2, 0.35, 0.5, 0.75, 1.0 on the source leg and 0.75 … 0.1 on the target leg.
  - The source scene stays readable in x̂₀ up to s ≈ 0.5.
  - The target animal appears on the target leg between s = 0.5 and 0.35. By s = 0.2 the output is essentially fixed.
  - x̂₀ at s ≥ 0.75 is noise, because the 1/α(s) factor (α(1) ≈ 0.0066) amplifies score error. That is expected and
    labelled.
- `steering_grid_big.png` (first 64 cats; only s* = 1, w = 0 is a coupling):

  | setting | s*=0.2 | 0.4 | 0.6 | 0.8 | **1.0 (valid)** | w=0.25 | w=0.5 | w=1.0 |
  |---|---|---|---|---|---|---|---|---|
  | cost (clamped) | 15.8 | 71.4 | 139.3 | 171.8 | **193.6** | 289.6 | 403.6 | 663.2 |
  | judge dog-rate | 42.2% | 64.1% | 73.4% | 79.7% | **78.1%** | 92.2% | 93.8% | 96.9% |

  (In iteration 04 the valid column was cost 117.2 at 81.2%. The cost rises with every steering step.)
- `best_pairs_big.png`, `best_pairs_dog_to_cat_big.png` use the strict rule P(target) ≥ 0.9 and P(source) ≤ 0.05. 307 /
  308 of 512 qualify. The judge still lets through several outputs that are visibly the source animal, or garbled
  (c→d #34, #450, #239 still look like cats; #387 and #84 are garbled). The same figures exist for M1-M4 and for FT
  (`best_pairs_<run>_big.png`; FT qualifies 329 / 300).
- `same_noise_both_models_big.png`: matched / mismatched cost on seed-0 noise is 0.082 for the iter05 pair (iter04
  0.071, FT 0.058). The iter-05 dog model gives almost the same images as the iter-04 dog model (0.007); the cat models
  are close too (0.032). Iter05 against the FT models is 0.047 / 0.049. **Same noise → same scene holds across all
  three pairs.**
- `compare_cat_to_dog_big.png`, `compare_dog_to_cat_big.png`: test images 0-31, with rows input / FT / iter04 / iter05 /
  M1 or M2 / M3 or M4.
- Lowest-p figures: `*_lowest_ptarget_big.png` for iter05 and M1-M4, plus `prior_*_lowest_ptarget_big.png` for every
  snapshot.

## 5. My by-eye counts (the validator's counts are authoritative)

Categories as before: (a) same scene with a convincing target animal, (b) still the source animal, (c) broken or new scene.
iter05, test images 0-31:

| | (a) | (b) | (c) |
|---|---|---|---|
| c→d | 21 (0w, 1, 2w, 3w, 6, 7, 8w, 13, 14, 15w, 16w, 19, 21w, 22, 24, 25, 26, 27, 29, 30w, 31) | 1 (9) | 10 (4, 5, 10, 11, 12, 17, 18, 20, 23, 28) |
| d→c | 9 (4w, 6, 13, 18, 21, 23, 26, 27, 29w) | 9 (0, 1, 2, 9, 14, 20, 22, 30, 31) | 14 (3, 5, 7, 8, 10, 11, 12, 15, 16, 17, 19, 24, 25, 28) |

Column by column, iter05 looks very much like iter04, which fits the same-noise numbers. It is higher in contrast in
c→d, with a few more broken columns. M1 / M2 rows are recognisable target animals in roughly the right pose but
blurred and colour-washed. M3 / M4 rows are posterised.

## 6. Success criteria (spec)

| criterion | target | result | met? |
|---|---|---|---|
| 1 prior KID cat / dog | ≤ 10 / ≤ 6 | 18.4 / 9.8 (picks); best of iter05 17.3 / 9.8 | **no** |
| 1 prior judge | 82% … ceiling + 2 | cat 84.2% ✓, dog 80.3% ✗ | no |
| 1 memo ≥ 0.8, saturated ≤ 0.5% | | memo 0.90 / 0.97 ✓; saturated 1.0% / 1.2% ✗ | no |
| 2 composed KID | ≤ 6 both | 16.9 / **5.98** | no (d→c yes) |
| 2 composed judge | ≥ 80% / 77% | 74.8% / 76.4% | no |
| 2 composed clamped cost | ≤ 132 / 140 | 222.7 / 160.4 | no |
| 3 eye (my count) | c→d ≥ 21 with ≤ 6 broken; d→c ≥ 16 with ≤ 8 broken | 21 with 10 broken / 9 with 14 broken | no |
| 4 replaces FT | KID ≤ 3.8 / 2.8, judge ≥ 76.8% / 76.6%, eye ≥ FT | no pair has KID ≤ 5.9 | **no** |

**Replacement rule: no pair, composed or mixed, replaces the fine-tuned pair.** The fine-tuned composed ODE remains
the best valid coupling: 101.6 / 104.7, judge 74.8% / 74.6%, KID 2.9 / 2.2.

**Most visually striking pairs** (all from-scratch ones are cat→dog unless noted):
- Still the FT pair overall.
- From scratch: iter05 c→d #6 (golden dog), #7 (white poodle), #13 (Boston terrier), #19 (fluffy white dog from an
  orange cat), #24, #31 (Pekingese), and best pairs #133, #208, #392, #331.
- d→c #26 and #27 (black cats), #21 (tabby), #23 (orange cat walking), and best pairs #307 (Pekingese → orange cat),
  #331 (bulldog → grey cat), #143.
- These are mostly the same scenes the validator picked in iteration 04, because the decoders are nearly identical on
  the same noise.

## 7. Notes for the validator / iteration 06

- The spec's branch "KID falls by < 15%" applies in the strong form: **KID rose by 25-50%**. The nf-64 nets were at
  their best around 60-80 train-min (the iteration-04 picks). More training, even at a decaying lr, hurts.
  Over-fitting on 5,000 images (the falling memo ratio of the 0.9995 EMA) is the likely limit, not lr noise.
- Mixed pairs need matched latent statistics. A deterministic per-family latent rescaling (for example matching
  per-frequency energy of FT latents to scratch latents) would still be a map. But its output law is not guaranteed,
  so it would have to be checked with KID like everything else.
- 235.0 GPU-min remain in run 2.
- **Volume paths (new, all under `/vol/iter05/`):**
  - EMA snapshots `{cat,dog}_{e9995,e9999}_s{step}.pth`: cat 28347 / 31247 / 34123, dog 28586 / 31505 / 34420.
  - Resumable states `{cls}_last_s{step}.pth` at each of those steps.
  - Picks: `iter05/cat_e9995_s31247.pth`, `iter05/dog_e9999_s28586.pth`.
  - Priors `prior_*.pt` (with Tweedie samples), latents `latents_{cat_to_dog,dog_to_cat}.pt`, outputs
    `transport_*_{iter05,M1..M4}.pt` (keys `s1.0`, `s1.0_tw`), and `transport_*_{FT,iter04}_tw.pt`.
  - `evalT_stats.json`, `evalT_small.pt`, `eval05.json`, `snapstats_{cat,dog}.json`, `memo_*.json`.
  - Smoke outputs under `/vol/iter05/smoke/`.
