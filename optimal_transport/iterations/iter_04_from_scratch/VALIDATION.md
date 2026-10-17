# Validation of iteration 04 (models trained from scratch), and the spec for iteration 05

Validator agent, 2026-10-16 ~19:30 PDT. No GPU used. I read RUN2_BRIEF, AGENT_CONTEXT, CLOUD_HANDOFF, PLAN.md
(with its addendum), REPORT.md, metrics.json, eval04.json, evalT_stats.json, steering_grid.json, snapstats,
gpu_log, BUDGET_RUN2.md, LOG.md, `tasks_iter04.py`, `launch_iter04.py`, `pf_ode.py`, and the iteration-01/02
VALIDATION files. I looked at every key figure, and also at side-by-side crops. For test images 0-31 in both
directions I set input / iteration-01 fine-tuned output / iteration-04 from-scratch output side by side (the
inputs are identical in both figures, which I checked), plus 8× zooms of the doubtful columns. Local work was
limited to cropping PNGs and a CPU parameter and timing count of wider U-Nets.

## 1. Visual counts (s* = 1, w = 0, the only valid coupling)

Categories as in iteration 01: (a) a convincing target animal in the same scene, pose and colours; (b) still
the source animal; (c) broken, blurry or garbled, or a target animal in a clearly new scene.

| figure (from scratch) | (a) same-scene | (b) unchanged | (c) broken / new scene |
|---|---|---|---|
| cat→dog 0-15 | **11** (0, 1, 2w, 3w, 4, 6, 7, 10, 13, 14, 15w) | 2 (8, 11) | 3 (5 large dog face replaces the window scene; 9 blur; 12 headless body on grass) |
| cat→dog 16-31 | 10 (16, 19, 21w, 22, 25, 26, 27, 29, 30w, 31) | 1 (24 siamese) | 5 (17 and 23 new composition; 18 deer-like; 20 blur; 28 pink blob) |
| dog→cat 0-15 | **4** (4, 6, 8, 13) | 6 (0, 1, 9, 11, 14, 15) | 6 (2, 3, 5, 7, 10, 12: smeared blobs) |
| dog→cat 16-31 | 6 (18, 21, 23, 26, 27, 31w) | 4 (20, 22, 24, 30) | 6 (16, 17, 19, 25, 28, 29) |

(w = weak but acceptable.)

| | (a) | (b) | (c) |
|---|---|---|---|
| **First 16 of each direction (32 columns), iteration 04** | **15 (47%)** | 8 (25%) | 9 (28%) |
| First 16 of each direction, iteration 01 (fine-tuned) | 18 (56%) | 11 (34%) | 3 (9%) |
| **All 64 columns, iteration 04** | **31 (48%)** | 13 (20%) | 20 (31%) |
| All 64 columns, iteration 01 (fine-tuned) | 37 (58%) | 19 (30%) | 8 (12%) |
| cat→dog only, 32 columns: iteration 04 / iteration 01 | 21 (66%) / 19 (59%) | 3 / 9 | 8 / 4 |
| dog→cat only, 32 columns: iteration 04 / iteration 01 | 10 (31%) / 18 (56%) | 10 / 10 | 12 / 4 |

**Reading.**
- **cat→dog is as good as the fine-tuned baseline by eye, and fails differently.** The from-scratch dog
  decoder leaves far fewer cats unchanged (3/32 against 9/32). The yawning cat (3), the orange cats (19, and
  the "Pekingese" in best pairs) and the cat on black (21) now become dogs. That is the purity gain the judge
  reports, and it is real. The price is more broken or blurry columns (8 against 4).
- **dog→cat is clearly worse.** The from-scratch cat model is the weak link (prior KID 14.1 against 8.2 for
  dog). About a third of its outputs are smeared blobs (2, 3, 5, 16, 17, 25). The judge still calls 77% of
  them cats, while by eye only ~31% of the first 32 are convincing cats. **The judge over-credits blurry
  outputs even more than in iteration 01.** For these models its target rate is not a trustworthy validity
  measure. KID, which is 3-4× worse than baseline, agrees with the eye.
- **Blur and sharpness:** yes, the from-scratch images are noticeably softer and lower in contrast than the
  fine-tuned ones, especially fur texture and faces. This shows directly in the same-noise figure, where rows 1-2
  (new) look washed out next to rows 3-4 (old). The KID gap shows up in the images.
- **Lowest-p figures.** The failures are unchanged sources (frog-coloured input, several cats or dogs) and
  garbled blobs. **No clear other-class objects** (boats, horses, trucks) appear, unlike the fine-tuned dog
  model's lowest-p figure in iteration 02. So training from scratch did remove the other-class modes. The
  remaining failures are quality failures.
- **Best pairs.** The rule over-credits here too. Of the 16 c→d "best pairs", about 8 are clearly striking
  (idx 121 Pomeranian, 300 dog with pink collar, 248 black-and-white dog face, 349 chihuahua face, 103 grey
  schnauzer from a grey long-haired cat, 133 Pekingese from an orange cat, 208 boxer face, 455 dog in the
  sink). About 5 still read as cats (217, 41, 221, 2, 214). Of the d→c best pairs, about 8 are good (331
  bulldog → grey cat, 469, 53, 143, 325, 499, ...), and 5-6 are brown or white smears (122, 40, 196, 307, 275).
- **Steering grid.** Correct and clearly labelled. The dog-rate saturates by s* = 0.6, and guidance redraws the
  scene (row 6: the window disappears into a white-dog portrait). It is diagnostic only, as stated.
- **Same noise → same scene: confirmed visually.** All four models (new cat, new dog, old cat, old dog) put
  essentially the same layout, palette and pose on each seed. This is the strongest scientific finding of the
  iteration. Independently trained single-class models agree on the scene, so the composed ODE does not need
  a shared initialisation, and encoders and decoders from different training runs can be mixed.
- **Flow grid.** As the report says, it is uninformative: clamped raw states at s ≥ 0.2 are colour noise.
  **One useful detail:** in row 6 (the window cat), the window bars are still visible in the latent at
  s = 1.0. The encoded latents keep low-frequency image structure, so they are not exactly N(0, I). See §2.

**Most striking pairs so far.**
- Fine-tuned baseline: c→d 2, 6, 7, 8, 11, 22, 27 and d→c 4, 12, 24, 26, 27, 30 (iteration-01 list).
- From scratch: c→d 6 (golden dog), 7 (white poodle), 13 (Boston terrier), 14 (beagle face), 19 (Pomeranian
  from an orange cat), 27 (chow), 29 (black dog by the fence), 31 (Pekingese), and best-pairs idx 103 and
  133; d→c 26 and 27 (black cats lying), 21 (tabby face from a springer), 8 (grey cat on snow), and
  best-pairs idx 331.

## 2. Number and bug checks

- **Budget:** 1.4 + 0.9 + 68.5 + 68.5 + 2.9 + 42.2 + 42.2 + 7.5 = **234.1**, which matches. calls.txt lists
  exactly the four post-recovery spawns. Run-2 total 234.1 / 600, **365.9 left**.
- **Comparability.** The fine-tuned baseline rows in `composed_ode` reproduce iteration 01 exactly (101.596 /
  104.681, 74.80% / 74.61%, KID 2.92 / 2.16), re-scored with the same judge, KID references and 512 inputs. So
  the two rows of the main table are comparable. Ceilings (85.1 / 87.7) and floors (0.09 / 0.36) are as
  before.
- **Prior statistics use a different protocol from the transports:** n = 256 and 100 Heun steps, against
  n = 512 and 200 steps for transports, and n = 512 and 200 steps for the fine-tuned priors of iteration 02.
  That is fine for ranking snapshots, but not for comparing with iteration 02's prior KIDs at better than
  ~1 se. Iteration 05 should use n = 512 and 200 steps for priors.
- **Pick rule:** applied correctly. But KID differences among the late snapshots (cat 14.1-14.5 with se 1.7,
  dog 8.2-8.9 with se 1.3) are pure noise, so the choice of s17297 / s20119 over the final snapshots is
  arbitrary. A tie-break is specified below. This does not change any conclusion.
- **Output overshoot / saturated samples.** Transports reach [−1.37, 1.47], with 1.8% / 3.7% of pixels
  outside [−1, 1] (fine-tuned: 0.5% / 0.9%). **Two of the 16 lowest-p prior cat samples and two of the dog
  samples are completely white tiles** (mean 255, std 0 after clamping). The small nets sometimes blow out to
  a saturated image. This is rare, but the next eval should count it (the fraction of samples with more than
  50% of pixels clipped). Clamping only lowers cost, and cost is reported clamped too (132.0 / 139.9), so the
  numbers are honest.
- **No final denoising step.** `pf_ode.decode` stops at s = eps = 1e-3 with residual noise of std ≈ 0.01 and
  no Tweedie step. Adding x̂₀ = (x + σ²(eps)·score(x, eps)) / α(eps) at the end is one network evaluation and
  a deterministic map, so T stays a valid map. It usually helps Inception-based metrics a little. Report the
  outputs both with and without it.
- **Latents are not exactly Gaussian, and this matters for validity.** T(dogs) through the cat model has KID
  8.6, while the same cat model's own prior samples have KID 14.1. The same holds for the fine-tuned pair
  (2.2 against 4.4). So **T's output law is not the decoder's prior law**: real-image low-frequency structure
  survives the encoder (as seen in the flow grid) and is carried into the outputs. That is not a problem in
  itself, because validity is checked directly on T(X) through judge and KID. But (i) prior-sample metrics
  are only a proxy, and (ii) any mixed encoder/decoder pair must have its Gaussianity checked, not assumed.
  Spec below.
- "Source class on outputs = 66/512" appears in three runs (c→d scratch, d→c scratch, d→c fine-tuned). I
  checked the histograms: they differ in every other entry, so it is a coincidence, not a copy bug.
- Steering-grid statistics are on the first 64 cats (valid column cost 117 against 132 on all 512). The first
  64 are an easier-than-average subset, as noted in iteration 01. It is labelled, and fine.
- Minor point: the REPORT summary says "140" for the d→c cost. The unclamped value is 141.3 and the clamped one
  139.9. The table is correct.
- No test leakage: training uses `cifar("train", cls)` only, and memo ratios are 0.96-1.11 with no sample
  within 10% of the test median. The memo ratio drifts down slowly with training (cat 1.11 → 1.02, dog
  1.07 → 0.96). Keep watching it.

## 3. Verdict

- **Best valid coupling so far, by the distribution-level checks: still the fine-tuned composed ODE**
  (iteration 01): cost 101.6 / 104.7, judge 74.8% / 74.6% (ceilings 87.7 / 85.1), KID×1e3 2.9 / 2.2 (floors
  0.36 / 0.09). The from-scratch pair has the better judge rate (80.1% / 77.3%), but the judge over-credits
  its blurry outputs. KID is 3-4× worse (9.2 / 8.6), and the cost is 30-35% higher. Neither is fully valid.
  The fine-tuned pair is closer on KID, which is the check independent of the judge.
- **Most visually striking pairs:** still the fine-tuned pair overall, because it is sharper and does well in
  both directions. **For cat→dog only, the from-scratch pair is now competitive by eye**: 66% against 59%
  convincing, with far fewer unchanged cats, but blurrier and with more broken columns. Its dog→cat direction
  is clearly worse (31% against 56%) because the from-scratch cat model is weak.
- **Diagnosis:** training from scratch fixed the other-class modes (no boats or horses in any lowest-p
  figure). What is left is the **sample quality** of 9.9M-parameter nets trained for 87 minutes at a constant
  lr with a short EMA. The KID plateau after ~60 minutes, with the memo ratio still near 1, fits under-fitting
  from lr noise or capacity, not memorisation.

## 4. Choice of iteration 05

| option | expected gain toward striking valid pairs | GPU | verdict |
|---|---|---|---|
| (a) wider net from scratch (nf 96: 22.2M params, ~1.8× slower; nf 128 with 2 res blocks: 39.4M, ~2.3× slower, measured on CPU) | Possibly large at convergence. But the nf-64 net needed ~15k steps (60 min) to plateau, and nf 96 at ~2.3 it/s needs ≥ 110 min per class to reach that many steps, ≥ 230 GPU-min for both. **It does not converge within ~170.** | 230+ | not now; conditional option for iteration 06 (dog only), below |
| (b) resume the small nets from `_last.pth` with **cosine lr decay + a longer EMA (0.9999)** | KID plateaued at constant lr 2e-4 with EMA 0.9995 (a horizon of ~2k steps ≈ 8 min). Lr decay plus a long EMA is the standard cure for a plateau caused by lr noise, and is cheap. Expect −20 to −40% KID. It is also the cleanest test of lr noise against capacity. | ~2 × 58 | **CHOSEN (core)** |
| (c) mixed encoder/decoder pairs, decoding saved latents only | Cheap. FT-cat latents → new dog decoder could combine purity with the baseline's lower cost. A scratch encoder → FT decoder tells which leg drives cost and sharpness. Validity requires checking latent Gaussianity. | ~14 | **CHOSEN (add-on)** |
| (d) flow grid of denoised estimates x̂₀, and a final Tweedie step | Required figure fix; free | ~1 | **CHOSEN (add-on)** |
| more guidance or partial encoding | not couplings | n/a | no |

## 5. SPEC: iteration 05 (`iter_05_decay`)

**Code.** New `optimal_transport/tasks_iter05.py` (`from tasks_iter04 import *`), `modal_iter05.py` (copy of
`modal_iter04.py`, app name `follmer-cat-dog-iter05`, A10G, timeout 2.4 h), and `launch_iter05.py` (copy of
`launch_iter04.py` with ITER_DIR → `iter_05_decay`). Outputs go only under `/vol/iter05/`. Never write to
`/vol/iter04/` (read only).

**Launch (robust, as iteration 04 learned):** `modal deploy optimal_transport/modal_iter05.py`, then
`python3 optimal_transport/launch_iter05.py spawn smoke`. Then run `fetch` under `caffeinate -is`, and spawn A
and B together (2 containers); after both finish, spawn `eval`. Never use a `modal run` client for long jobs.
Redeploy after any code change.

### New function `resume_decay(cls, last, folder="iter05", cap_min=45.0, lr_max=2e-4, lr_min=2e-6, bs=256, ema_rates=(0.9995, 0.9999), snap_fracs=(0.5, 0.75, 1.0), seed=5, max_steps=None, vol=None)`
- `last = f"iter04/{cls}_last.pth"` (step 22,457 cat / 22,697 dog). Build the net with `build_network(REPO,
  CONFIG, dev, ck["config_overrides"])`, load the **raw** weights `ck["model"]`, load Adam state
  `ck["optimizer"]` (if it fails, use a fresh Adam with a 200-step warmup and log it). Initialise **both** EMA
  trackers from `ck["ema"]["shadow_params"]` (the 0.9995 EMA).
- The data and loss are exactly as in iteration 04: CIFAR-train class only, hflip, VP continuous loss, bf16,
  grad clip 1.0, `s*999` label.
- **LR is time-based cosine:** lr = lr_min + ½(lr_max − lr_min)(1 + cos(π·p)), with p = elapsed / cap,
  checked every step, with no warmup if Adam state loaded. EMA decays are fixed at 0.9995 and at 0.9999.
- At 50/75/100% of the cap, save both EMAs as `/vol/iter05/{cls}_e9995_s{step}.pth` and
  `/vol/iter05/{cls}_e9999_s{step}.pth` (with `config_overrides`, meta lists lr schedule and EMA). **At every
  snapshot, also save the resumable state** `/vol/iter05/{cls}_last_s{step}.pth` (raw weights, both EMAs,
  Adam), so a cancellation never loses state again. `vol.commit()` after each save.
- Then `snap_eval` on the 50%-0.9999, 75% (both) and 100% (both) snapshots, plus the iteration-04 pick as a
  reference: **n = 512, `prior_z(512)` (seed 0), 200 Heun steps, eps 1e-3**, judge, KID vs all 1,000 test
  images of the class, memo check. That is the transport protocol, so the numbers compare directly. Also
  report `frac_samples_saturated` (samples with more than 50% of pixels outside [−1, 1]) and the KID with the
  final Tweedie step.
- **Pick rule:** discard memo ratio < 0.5, which is a hard flag (warn below 0.8). Take the lowest KID. If
  several are within 1 se of the lowest, take the one with the highest judge rate that does not exceed the
  ceiling + 2 points. If there is still a tie, take the latest step.

### Chains (A10G each)
- `smoke` (1 container, ~3 min): `resume_decay("cat", ..., max_steps=20, folder="iter05/smoke")` loads
  `_last.pth`, with both EMAs, the cosine lr and both save types. Then `snap_eval` with n = 16 and 5 steps,
  and an `eval05` run with n = 8 and 10 steps that exercises every figure and the mixed and Gaussianity code.
- **A (cat):** `resume_decay("cat", "iter04/cat_last.pth", cap_min=45)` + snap_eval. ≈ 45 + 11 + 2 = **58 min**.
- **B (dog):** the same for dog. ≈ **58 min**. Run in parallel with A.
- **eval** (after A and B; ≈ 22 min):
  1. Composed ODE both directions with the new picks: 512 + 512, 200 Heun, eps 1e-3. Save the latents to
     `/vol/iter05/latents_*.pt`. Run the standard `eval_runs` (cost unclamped and clamped ± se, judge target
     vs ceiling, source rate, KID vs floor, lowest-p figure, overshoot and saturated fraction). Report each
     output set with and without the final Tweedie step (no extra ODE).
  2. **Mixed pairs** (decode only, from saved latents; each is a valid map with law ≈ decoder law, *if* the
     latents pass the Gaussianity check):
     - M1: `/vol/iter02/latents_cat_to_dog.pt` (FT cat encoder) → new dog decoder;
     - M2: `/vol/iter02/latents_dog_to_cat.pt` (FT dog encoder) → new cat decoder;
     - M3: new scratch-cat latents → FT dog decoder `out/dog.pth`;
     - M4: new scratch-dog latents → FT cat decoder `out/cat.pth`.
     Run the same eval on each. M3 and M4 have the same output law as the baseline, so they isolate the
     effect of the encoder on cost.
  3. **Gaussianity of every latent set** (iter02 FT, iter04 scratch, iter05 new, and a fresh N(0, I) sample
     of the same size as reference): overall mean and std, the distribution of per-image ‖z‖²/d, the
     energy ratio after 4×4 and 8×8 average pooling against the iid expectation (it should be ≈ 1), lag-1
     spatial autocorrelation, and the KS statistic of the pixel values against N(0, 1). Flag a latent set if
     its pooled energy ratio deviates from N(0, I) by more than 3× its own bootstrap se.
  4. Required figures for the picked pair, 4× NEAREST:
     - **Flow grid** showing the Tweedie estimate x̂₀(x_s) = (x_s + σ_s²·score) / α_s, not raw states, at
       s ∈ {0.1, 0.2, 0.35, 0.5, 0.75, 1.0} on the cat leg and {0.75, 0.5, 0.35, 0.2, 0.1} on the dog leg,
       using the nearest Heun grid points, plus input and output. Optionally add a second row per image
       with x_s/α_s, clamped. Both directions.
     - Steering grid, as in iteration 04.
     - Best pairs, with the stricter rule p_target(out) ≥ 0.9 **and** p_source(out) ≤ 0.05, cost ascending.
     - Same-noise figure.
     - **One comparison figure per direction for test images 0-31: rows input / FT baseline / iter04 / iter05
       / best mixed pair**, all from saved outputs. This is the figure the validator counts by eye.

**GPU estimate:** smoke 3 + A 58 + B 58 + eval 22 ≈ **141 GPU-min**, which brings run 2 to ≈ 375 / 600 and
leaves ≈ 225. Start nothing that would push the total past 600.

### Success criteria (valid coupling only: s* = 1, w = 0; each direction and each pair judged separately)
1. **Prior quality** at n = 512 and 200 steps: KID×1e3 ≤ **10** for cat (from 14.1) and ≤ **6** for dog (from
   8.2), judge between 82% and ceiling + 2, memo ratio ≥ 0.8, saturated samples ≤ 0.5%.
2. **Composed iter05 pair:** KID ≤ **6** in both directions (from 9.2 / 8.6), judge ≥ 80% / 77% (no loss),
   clamped cost ≤ 132 / 140 (no increase).
3. **By eye** (comparison figure, 32 columns each): c→d ≥ 21 same-scene with ≤ 6 broken, and d→c ≥ 16
   same-scene (from 10) with ≤ 8 broken.
4. **A pair becomes the new best valid coupling** only if its KID ≤ FT KID + 1 se (≈ 3.8 c→d / ≈ 2.8 d→c),
   its judge rate ≥ FT + 2 points, and the eye count is at least FT's (59% / 56%). The mixed pairs and their
   latent Gaussianity are judged by the same rule. Otherwise, report a trade-off.

### Fallbacks / decision for iteration 06 (≤ ~120 GPU-min, then a ≤ 25-min final eval)
- If the Adam state does not load: use a fresh Adam with a 200-step warmup (as in iteration 04), everything
  else unchanged.
- If a container is cancelled, the `_last_s*` and snapshot files exist. Evaluate what exists in a short job.
- **If lr decay cuts prior KID by ≥ 25%** but criterion 4 still fails: in iteration 06, resume again from
  `iter05/{cls}_last_s*.pth` for 40 min at low lr (5e-5 → 1e-6, EMA 0.9999), cat first, because cat is the weak
  direction.
- **If KID falls by < 15% (capacity-limited):** in iteration 06, train a **dog-only nf = 96, num_res_blocks = 2**
  net (22.2M params) from scratch for ~100 min (warmup 1k, constant 2e-4 to 60%, cosine to 2e-6, EMA
  min(0.9999, (1+t)/(10+t)), `_last` at each snapshot). Use it as the c→d decoder with the best Gaussian
  encoder. That is ≈ 115 GPU-min.
- **If nothing beats FT on criterion 4:** the final answer is the fine-tuned composed ODE (101.6 / 104.7,
  74.8% / 74.6%, KID 2.9 / 2.2). The final eval (≤ 25 min) re-runs it with the Tweedie step on all 1,000 test
  cats and dogs, and the gallery shows the best FT pairs next to the best from-scratch c→d pairs.
