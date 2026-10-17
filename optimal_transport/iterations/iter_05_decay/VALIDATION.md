# Validation of iteration 05 (lr-decay resume, mixed pairs, Tweedie step), and the stop/go call

Validator agent, 2026-10-16. No GPU used. I read RUN2_BRIEF, REPORT.md, PLAN.md with its addendum,
metrics.json, eval05.json, evalT_stats.json, cost_quantiles.json, snapstats, gpu_log, BUDGET_RUN2.md, LOG.md,
`iter_04_from_scratch/VALIDATION.md` and the relevant parts of `tasks_iter05.py` (Tweedie, Gaussianity, the
composed-ODE and mixed-pair loops). I looked at every key figure. For both compare figures I cut the 32
columns into 8-column panels at native resolution (rows input / FT / iter04 / iter05) and zoomed the
candidate headline pairs. Local work was PNG cropping and reading JSON only.

## 1. By-eye counts on the compare figures (test images 0-31, s* = 1, w = 0)

Categories as before: (a) a convincing target animal in the same scene; (b) still the source animal;
(c) broken, garbled or deer-like, or a target animal in a clearly new composition. w = weak but accepted.

**cat → dog** (`compare_cat_to_dog_big.png`)

| model | (a) same-scene | (b) unchanged | (c) broken / new scene |
|---|---|---|---|
| FT (iter01) | **20** (1, 2w, 4, 6, 7, 8, 10, 11w, 12w, 14w, 16, 19w, 20w, 22, 25w, 26w, 27, 29, 30, 31) | 8 (0, 3, 9, 13, 15, 21, 24, 28) | 4 (5, 17, 18, 23) |
| iter04 | **21** (0w, 1, 2w, 3w, 4w, 6, 7, 10, 13, 14w, 15w, 16, 19, 21w, 22, 25w, 26, 27, 29, 30w, 31) | 3 (8, 11, 24) | 8 (5, 9, 12, 17, 18, 20, 23, 28) |
| iter05 | **23** (0w, 1, 2w, 3w, 6, 7, 9w, 11w, 13, 14w, 15w, 16w, 19, 20w, 21w, 22, 24w, 25, 26, 27, 29, 30w, 31) | 1 (8) | 8 (4, 5, 10, 12, 17, 18, 23, 28) |

**dog → cat** (`compare_dog_to_cat_big.png`)

| model | (a) same-scene | (b) unchanged | (c) broken / new scene |
|---|---|---|---|
| FT (iter01) | **22** (0, 1w, 3w, 4, 5, 6w, 8w, 10, 12, 14, 18, 19w, 20w, 23, 24, 25, 26, 27, 28w, 29w, 30, 31w) | 7 (2, 9, 11, 13, 15, 21, 22) | 3 (7, 16, 17) |
| iter04 | 11 (4, 6, 8w, 13, 18, 21, 23, 26, 27, 29w, 31w) | 11 (0, 1, 2, 9, 11, 14, 15, 20, 22, 24, 30) | 10 (3, 5, 7, 10, 12, 16, 17, 19, 25, 28) |
| iter05 | 11 (4w, 6, 13w, 18, 21w, 23, 24w, 26, 27, 29w, 31w) | 8 (0, 1, 2, 9, 14, 20, 22, 30) | 13 (3, 5, 7, 8, 10, 11, 12, 15, 16, 17, 19, 25, 28) |

**All 64 columns:** FT 42 / 15 / 7 (66% / 23% / 11%); iter04 32 / 14 / 18 (50% / 22% / 28%);
iter05 34 / 9 / 21 (53% / 14% / 33%).

Notes.
- My counts are within 1-2 columns of the predecessor's for FT and iter04 (I was slightly more generous to FT
  d→c: 22 against 18). They agree with the reporter's iter05 counts within 2-3 columns per direction.
- **cat → dog from scratch is the one place where scratch matches or beats FT by eye** (23 against 20,
  with 1 unchanged cat against 8), at the price of twice as many broken columns (8 against 4) and softer
  textures. That replicates iteration 04.
- **dog → cat from scratch is still clearly worse** (11 against 22). Its failures are smeared blobs
  (3, 5, 10, 12, 16, 17, 25) and unchanged dogs. The decayed cat decoder did not fix that.
- iter05 and iter04 are almost column-for-column the same image (as the same-noise numbers predict).
  iter05 c→d has higher contrast and a few more "pops" (#9, #20, #24, #25), and a few more garbled ones (#4, #10).
- Mixed rows: M1/M2 are recognisable but washed out and blurred; M3/M4 are posterised and over-saturated,
  often a new scene. None is usable for a gallery.
- Strict best-pairs rule: still over-credits. In `best_pairs_big.png` (iter05 c→d) #34, #239, #450 are still
  cats and #387, #84 are garbled, as the reporter says. FT's low-cost best pairs (c ≈ 15-29) are mostly
  near-identity edits of inputs that already look dog-like, so they are *valid* but not *striking*.

## 2. The most visually striking cat ↔ dog pairs (across all iterations)

| # | direction | model | test idx | what | figure (row) |
|---|---|---|---|---|---|
| 1 | c→d | FT | 10 | tabby face → brown puppy face, same pose and background | `iter_05_decay/compare_cat_to_dog_big.png` (FT row) |
| 2 | c→d | FT | 8 | black-and-white cat → Pekingese, same red background | same, FT row |
| 3 | c→d | iter05 (iter04 almost identical) | 13 | black cat on a tan floor → Boston terrier | same, iter05 row |
| 4 | c→d | iter05 | 19 | orange long-haired cat → white fluffy dog | same, iter05 row |
| 5 | c→d | iter05 (iter04, FT similar) | 31 | dark cat silhouette on beige → Japanese chin / Pekingese | same, iter05 row |
| 6 | c→d | iter05 (iter04 similar) | 6 | cat on a blanket → golden retriever pup | same, iter05 row |
| 7 | c→d | iter05 | 25 | white cat on black → white bulldog | same, iter05 row |
| 8 | c→d | iter05 | 133 | orange cat → red-brown boxer face (iter04 #133: Pekingese) | `iter_05_decay/best_pairs_big.png`; `iter_04_from_scratch/best_pairs_big.png` (c=47) |
| 9 | c→d | iter05 | 392 | sitting brown cat → sitting brown dog, same yellow wall | `iter_05_decay/best_pairs_big.png` |
| 10 | d→c | FT | 4 | shepherd face → green-eyed cat face | `iter_05_decay/compare_dog_to_cat_big.png` (FT row) |
| 11 | d→c | FT | 14 | chihuahua in a red hood → cat face, hood kept as a red frame | same, FT row |
| 12 | d→c | iter05 | 331 | frontal dog → frontal grey tabby (iter04 has the same pair, c=30) | `iter_05_decay/best_pairs_dog_to_cat_big.png` |

Runners-up: FT d→c #25 (Pekingese → tabby, pose changes), FT d→c #12 (pug → tabby face), iter05 d→c #6
(black dog among pink flowers → black cat), iter05 d→c #307 and FT d→c #307 (Pekingese → orange cat),
iter04 c→d #103 (grey long-haired cat → schnauzer, `iter_04_from_scratch/best_pairs_big.png` c=46).

Split: c→d striking pairs are now mostly from-scratch; d→c striking pairs are mostly FT.

## 3. Numbers and bugs

- **Budget**: 2.2 + 54.2 + 54.0 + 20.5 = 130.9; 234.1 + 130.9 = 365.0. Matches gpu_log container minutes
  (+0.5 startup each). Correct.
- **Report tables match metrics.json/evalT_stats.json** (costs, clamped costs, judge, KID ± se, histograms,
  saturated fractions, cost quantiles). FT and iter04 rows reproduce the iteration-01/04 numbers.
  iter05 c→d judge = FT judge = 0.748 is a coincidence (different histograms).
- **Pick rule.** Implemented as specified. It cannot see the thing that mattered: the encoder. The cat pick
  (EMA 0.9995, s31247, memo 0.90) has the least Gaussian latents (pool8 1.40, norm. 1.56), and c→d cost
  jumped 132 → 223 with almost the same dog decoder (iter05 vs iter04 dog on same noise: 0.007).
  **Lesson: picking a checkpoint by prior KID/judge says nothing about its encoder. A pair must be picked on
  a small composed-transport probe (e.g. 64 images: cost + KID), not on prior stats.**
- **Tweedie step**: code is correct, (x + σ²·score)/α at s = eps. It worsens KID in 10/10 sets by about 1 se
  each; the consistent sign makes the conclusion (do not adopt) sound.
- **Gaussianity code** is correct (k²·E[avgpool_k(z)²] is 1 for iid N(0,1); lag-1 normalised by the set's
  energy). Findings stand: no encoder gives N(0, I); all carry 25-56% excess low-frequency energy.
- **Validity caveat (already noted in iteration 04, now sharper):** iter05 T(dogs) has KID 5.98 while the cat
  model's own prior has KID 18.4. The encoded latents are not N(0, I), so the composed map's output law is
  not the decoder's prior law; here it is *closer to real cats* than the prior. The "valid coupling" label
  is approximate for every pair, FT included. KID on T(x) is the right test of the target marginal, and that
  is what the report uses.
- **Deviation** (M5/M6 not run because the spawn used the old deployment): honestly reported; the argument
  that they would repeat M1/M2 is sound given the 0.007 / 0.032 same-noise distances.
- Minor: `same_noise` compares iter05 200-step decodes with iter04 100-step decodes; the 0.032 cat distance
  includes some step-count difference. Not material.
- No other bugs found.

## 4. Recommendation: **STOP** (go to the final report)

Reasons.
1. **Nothing plausible beats FT on the replacement rule within 120 GPU-min.** FT is a 61.8M-param net
   pretrained on all 50k CIFAR images; its composed KID is 2.9 / 2.2. The best from-scratch decoders reach
   prior KID 13.7 / 8.75 (cat / dog) and peak at 60-80 train-min, then over-fit on 5,000 images
   (iteration 05 proved more training makes them worse). A wider nf = 96 net (22M params) on the same
   5,000 images will over-fit sooner, not later. I put **P(KID ≤ FT + 1 se in c→d) ≈ 5%**.
2. **The mixed-pair fix cannot win either.** Rescaling or whitening FT latents to unit energy before decoding
   with a scratch decoder is still a deterministic map, so (x, T(x)) is still a coupling. But its output law
   is bounded by the scratch decoder's quality (KID ≥ ~6-9 at best) and only the total energy is fixed, not
   the 25-30% low-frequency excess every encoder carries. **P(beats FT) ≈ 2%; P(matches iter04 KID ≈ 9) ≈ 25%.**
   It would not produce more striking pairs than the composed scratch pairs already in hand.
3. **The visual goal is already served.** The striking c→d pairs come from the scratch models (iter04/05),
   the striking d→c pairs from FT, and both sets exist as saved outputs. A sixth iteration would most
   likely reshuffle the same scenes, because every model maps the same noise to the same scene.

If the user still wants to spend: the only iteration-06 I would run is **(A) an encoder-aware re-pick, no
training, ≈ 30 GPU-min**: decode with every saved scratch snapshot pair on a 64-image probe (cost + KID +
eye), then run the best encoder/decoder pair at n = 512. Success: c→d KID ≤ 7 with cost ≤ 125 and eye ≥ 23/32;
d→c KID ≤ 5 with cost ≤ 140 and eye ≥ 14/32. P(success in at least one direction) ≈ 35%; P(it beats FT) ≈ 0.
The fallback in the iteration-04 spec (nf = 96 dog net, ~115 GPU-min) I do **not** recommend, for reason 1.

### What should headline the final report
- **Best valid coupling: the FT composed ODE**, cost 101.6 / 104.7, judge 74.8% / 74.6% (ceilings 87.7 /
  85.1%), KID 2.9 / 2.2, eye 66% same-scene over 64 columns. No Tweedie step.
- **Scientific result: same noise → same scene across independently trained models** (`iter_05_decay/same_noise_both_models_big.png`,
  matched / mismatched 0.06-0.08 for all pairs; 0.007 between dog snapshots). The composed ODE
  works between models that never shared weights or data.
- **Trade-off result: from-scratch c→d** (iter04: cost 132, judge 80.1%, KID 9.2) leaves 1-3 cats unchanged
  out of 32 against FT's 8, and the off-class modes (boats, horses) are gone, at the price of softer images
  and more broken columns. From-scratch d→c is not competitive.
- **Gallery:** the 12 pairs in §2. Figures: `iter_05_decay/compare_cat_to_dog_big.png`,
  `iter_05_decay/compare_dog_to_cat_big.png` (input / FT / iter04 / iter05 side by side),
  `iter_05_decay/best_pairs_big.png`, `iter_05_decay/best_pairs_dog_to_cat_FT_big.png`,
  `iter_05_decay/best_pairs_dog_to_cat_big.png`, and one flow grid (`iter_05_decay/flow_grid_big.png`,
  now showing denoised estimates).
- **Negative results worth stating:** guidance (iter02) is not a coupling and costs 2-4×; CINIC and CIFAR
  re-fine-tunes did not help; lr decay on the small nets made them worse; mixed encoder/decoder families fail
  from a latent-statistics mismatch; the final Tweedie step hurts KID.
- **Final eval:** the report can be built from saved outputs at 0 GPU-min. If 1,000-image numbers are
  wanted, run the FT composed ODE on the remaining 488 + 488 test images (≈ 24 GPU-min, no Tweedie).
  Run 2 would end at ≈ 389 / 600.
