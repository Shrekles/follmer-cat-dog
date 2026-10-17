# Iteration 06 report: FT pair on the full test set, and an encoder-aware re-pick of the from-scratch snapshots

Training agent, 2026-10-17. No training. Plan: `PLAN.md` (the selection rule was written there before running).
Code: `optimal_transport/tasks_iter06.py`, `modal_iter06.py`, `launch_iter06.py`. Deployed app
`follmer-cat-dog-iter06`, spawned and fetched under `caffeinate -is`, then stopped. Numbers come from `ft1000.json`,
`repick06.json`, `evalT_stats.json` and `memo_T_*.json`, collected in `metrics.json`.

**Summary.**
- **FT pair on all 1,000 test images:**
  - cat→dog: cost 99.1 ± 2.2, judge 73.6%, KID 2.82.
  - dog→cat: cost 108.9 ± 2.7, judge 76.8%, KID 2.40.
  - The 0-511 subset reproduces 101.6 / 104.7 exactly, and re-running 16 images matches the stored outputs to 1e-3.
- **Re-pick:** encoder quality falls steadily with training. In every case the earliest eligible snapshot (s12k) is
  the best encoder: lowest cost, lowest KID, and the most Gaussian latents.
  - The selected cat→dog pair (cat_s12129 → dog_s20119) beats the iter04 pick on cost (123.2 vs 132.0) and on KID
    (8.3 vs 9.2), with the same judge rate.
  - The selected dog→cat pair (dog_s17541 → cat_s22457) almost halves iter04's KID (4.66 vs 8.62) at slightly
    lower cost (136.8 vs 139.9).
- **Success criteria:**
  - cat→dog: not met (KID 8.3 > 7).
  - dog→cat: KID and cost are met, but my strict by-eye count is 13/32 against the required 14. It is 14 if one
    borderline column is accepted.
- **FT remains the best valid coupling.**

## GPU use

| job | container min | + startup | GPU-min |
|---|---|---|---|
| smoke 1 (failed at the first figure: an empty 16-col grid at n = 8; fixed) | 0.34 | 0.5 | 0.8 |
| smoke 2 (all code paths, tiny sizes) | 0.91 | 0.5 | 1.4 |
| A: FT on 488 + 488 and 1,000-image eval | 19.82 | 0.5 | 20.3 |
| B: re-pick stages 1-3 and figures | 24.11 | 0.5 | 24.6 |
| **iteration total** | | | **47.1** (cap 60) |

Run-2 total: **412.1 / 600**.

## A. FT composed ODE on the full CIFAR-10 test set

- Setup: Heun with 200 steps per leg, eps = 1e-3, no Tweedie step.
- Ceilings on all 1,000 real test images: dog 87.7%, cat 85.1%.
- KID ×1e3 is measured against all 1,000 test images of the target class. Floors (1,000 train vs 1,000 test images):
  dog 0.06 ± 0.22, cat −0.09 ± 0.17.

| direction | images | cost | cost clamped (median) | judge target | judge source on T(x) | KID ×1e3 |
|---|---|---|---|---|---|---|
| c→d | **all 1,000** | **99.1 ± 2.2** | 99.1 (77.4) | **73.6 ± 1.4%** | 16.5% | **2.82 ± 0.71** |
| c→d | 0-511 (iter01 file) | 101.6 ± 3.4 | 101.6 (76.3) | 74.8% | 16.2% | 2.92 ± 0.92 |
| c→d | 512-999 (new) | 96.6 ± 2.9 | 96.6 (78.6) | 72.3% | 16.8% | 2.75 ± 0.90 |
| d→c | **all 1,000** | **108.9 ± 2.7** | 108.9 (85.1) | **76.8 ± 1.3%** | 10.8% | **2.40 ± 0.50** |
| d→c | 0-511 (iter01 file) | 104.7 ± 3.4 | 104.6 (85.4) | 74.6% | 12.9% | 2.16 ± 0.64 |
| d→c | 512-999 (new) | 113.4 ± 4.3 | 113.3 (84.0) | 79.1% | 8.6% | 2.69 ± 0.65 |

- **Reproduction check:** the 0-511 rows are the iteration-01 outputs (`/vol/iter01/transport_cat_to_dog_B.pt`,
  `transport_dog_to_cat_A.pt`). They give 101.6 / 104.7 (cost) and 74.8% / 74.6% (judge).
- **Determinism check:** re-transporting the first 16 images with the iteration-06 code gives max |Δ| = 1.2e-3 and
  1.1e-3, and cost 76.52 vs 76.52 and 103.16 vs 103.16. So the 512-999 outputs come from the same map.
- The two halves agree within noise. The d→c cost on 512-999 is 113 against 105, a difference of about 1.6 se of the
  difference.
- Saturated outputs: 0 (c→d) and 1 of 1,000 (d→c).
- Strict best pairs: 638 / 1,000 qualify for c→d and 618 / 1,000 for d→c.
- Figures:
  - `ft_{cat_to_dog,dog_to_cat}_512_big.png` and `_512_b_big.png` (test images 512-543);
  - `ft_*_1000_lowest_ptarget_big.png`;
  - `ft_best_pairs_*_1000_big.png`.

## B. Encoder-aware re-pick

All results here use the first 64 test images of the source class.

### Stage 1: every encoder with a fixed decoder

- Columns: clamped cost ± se, judge target rate, KID ×1e3 (diagnostic only at n = 64), and latent statistics:
  E|z|²/d and the normalised pooled-8 energy ratio (both 1 for N(0, I)).
- **Bold** marks the reference iter04 pair, which sets the cost cap.

**cat→dog** (decoder iter04 `dog_s20119`; cap = 117.25):

| encoder | cost cl. | judge dog | KID | E\|z\|²/d | pool8 norm. |
|---|---|---|---|---|---|
| 4_cat_s12129 | **108.0 ± 6.6** | **82.8%** | **7.6** | 0.953 | **1.19** |
| 4_cat_s14714 | 113.7 | 81.2% | 8.1 | 0.953 | 1.21 |
| **4_cat_s17297 (iter04 pick)** | **117.2** | 81.2% | 10.8 | 0.943 | 1.24 |
| 4_cat_s19874 | 127.9 | 82.8% | 14.1 | 0.936 | 1.27 |
| 4_cat_s22457 | 144.3 | 81.2% | 14.3 | 0.932 | 1.30 |
| 5_cat_e9999_s28347 / s31247 / s34123 | 149.6 / 153.2 / 160.3 | 78-83% | 16-19 | 0.90-0.92 | 1.33-1.42 |
| 5_cat_e9995_s28347 / s31247 / s34123 | 181.3 / 203.9 / 217.6 | 78-80% | 16-23 | 0.90-0.92 | 1.42-1.59 |

**dog→cat** (decoder iter04 `cat_s17297`; cap = 136.51):

| encoder | cost cl. | judge cat | KID | E\|z\|²/d | pool8 norm. |
|---|---|---|---|---|---|
| 4_dog_s12386 | **111.1 ± 7.4** | **89.1%** | **3.4** | 0.947 | **1.21** |
| 4_dog_s14961 | 119.3 | 81.2% | 4.8 | 0.950 | 1.24 |
| 4_dog_s17541 | 123.7 | 87.5% | 6.6 | 0.939 | 1.28 |
| **4_dog_s20119 (iter04 pick)** | **136.5** | 79.7% | 8.6 | 0.931 | 1.32 |
| 4_dog_s22697 | 154.8 | 79.7% | 12.0 | 0.927 | 1.36 |
| 5_dog_e9999_s28586 / s31505 / s34420 | 166.0 / 180.5 / 202.6 | 77-80% | 14-23 | 0.89-0.91 | 1.41-1.52 |
| 5_dog_e9995_s28586 / s31505 / s34420 | 227.6 / 284.6 / 323.7 | 67-75% | 21-34 | 0.89-0.92 | 1.51-1.70 |

**Finding: encoder quality falls with training.** For both classes and both EMAs:
- cost, KID and the latents' excess low-frequency energy rise steadily with the training step;
- the 0.9995 EMA of iteration 05 is the worst encoder of all.

This confirms the validator's diagnosis: picking by prior samples alone cannot see this. The iter04 picks (chosen by
prior KID) were the third and fourth best encoders out of 11.

### Stage 2: top-3 encoders × all 11 decoders

The decoder matters less than the encoder: for a fixed encoder, cost varies by about 20-30 across decoders.

Best pairs under the rule (judge rate first, at cost ≤ cap):

| direction | pair | cost cl. | judge | KID (n = 64) |
|---|---|---|---|---|
| c→d | **4_cat_s12129 → 4_dog_s20119 (selected)** | 108.0 | 82.8% | 7.6 |
| c→d | 4_cat_s17297 → 4_dog_s17541 | 111.6 | 82.8% | 9.4 |
| c→d | 4_cat_s12129 → 4_dog_s14961 | 95.5 | 81.2% | 7.8 |
| c→d | 4_cat_s14714 → 4_dog_s22697 | 117.1 | 78.1% | 6.0 |
| d→c | **4_dog_s17541 → 4_cat_s22457 (selected)** | 136.3 | 92.2% | 5.8 |
| d→c | 4_dog_s12386 → 4_cat_s17297 | 111.1 | 89.1% | 3.4 |
| d→c | 4_dog_s12386 → 4_cat_s19874 | 120.2 | 89.1% | 4.5 |
| d→c | 4_dog_s12386 → 4_cat_s12129 | 100.1 | 85.9% | 3.6 |

**Caveat on the rule.** It maximises the judge rate up to the cost cap. In d→c it therefore chose a pair whose probe
cost (136.3) sits just under the cap. 4_dog_s12386 → 4_cat_s17297 was one judge step lower (89.1% against 92.2%,
2 images out of 64) and 25 cheaper, with a better probe KID. That pair was not run at n = 512 (the rule was fixed in
advance), so its full-set numbers are unknown. It is the obvious candidate if anyone continues.

### Stage 3: selected pairs on the first 512 test images

The FT and iter04 rows are re-scored from their saved outputs in the same call.

KID floors (512 train vs 1,000 test): dog 0.36, cat 0.09.

| direction | coupling | cost | cost clamped | judge target (ceiling) | src on T(x) | KID ×1e3 | memo ratio of T(x) |
|---|---|---|---|---|---|---|---|
| c→d | FT (iter01) | 101.6 ± 3.4 | **101.6** | 74.8% (87.7%) | 16.2% | **2.92 ± 0.92** | 1.02 |
| c→d | iter04 pick | 132.4 ± 3.1 | 132.0 | 80.1% | 12.9% | 9.22 ± 1.35 | 1.16 |
| c→d | **re-pick 4_cat_s12129 → 4_dog_s20119** | 123.4 ± 3.1 | 123.2 | **81.1%** | 11.3% | 8.34 ± 1.26 | 1.08 |
| d→c | FT (iter01) | 104.7 ± 3.4 | **104.6** | 74.6% (85.1%) | 12.9% | **2.16 ± 0.64** | 1.16 |
| d→c | iter04 pick | 141.3 ± 3.2 | 139.9 | 77.3% | 12.9% | 8.62 ± 1.06 | 1.28 |
| d→c | **re-pick 4_dog_s17541 → 4_cat_s22457** | 137.2 ± 2.9 | 136.8 | **78.3%** | 11.1% | 4.66 ± 0.77 | 1.18 |

**Held-out images 64-511 only** (n = 448; the 64 probe images were used for selection):

| direction | coupling | cost clamped | judge | KID |
|---|---|---|---|---|
| c→d | FT | 103.7 ± 3.7 | 74.6% | 3.09 ± 0.98 |
| c→d | iter04 | 134.1 ± 3.4 | 79.9% | 9.23 ± 1.36 |
| c→d | re-pick | 125.4 ± 3.4 | 80.8% | 8.60 ± 1.31 |
| d→c | FT | 104.8 ± 3.6 | 74.1% | 2.12 ± 0.71 |
| d→c | iter04 | 140.4 ± 3.3 | 77.0% | 8.60 ± 1.22 |
| d→c | re-pick | 136.9 ± 3.0 | 76.3% | 4.47 ± 0.83 |

- The held-out numbers match the full-512 numbers, so selection bias is small.
- The re-pick's probe-set advantage in judge rate (92% for d→c) did not carry over to the held-out images (76.3%).
  The KID and cost gains did.

**Memorisation.**
- "memo ratio" is the median nearest-CIFAR-train distance of T(x) divided by that of real test images. All ratios
  are ≥ 1.02, so there is no sign of memorisation.
- The snapshots' own prior memo ratios, from iterations 04 and 05:

  | snapshot | memo ratio |
  |---|---|
  | cat_s12129 | 1.11 |
  | dog_s20119 | 0.99 |
  | dog_s17541 | 0.99 |
  | cat_s22457 | 1.02 |

**Saturated outputs:** pixels outside [−1, 1] make up 0.95% (c→d) and 1.98% (d→c) of the re-pick outputs, against
1.8% / 3.7% for iter04.

### By-eye counts (my own) on `compare_*_big.png`, re-pick row, test images 0-31

Categories: (a) same scene with a convincing target animal (w = weak but accepted); (b) still the source animal;
(c) broken, garbled, or a target animal in a clearly new composition.

| direction | (a) | (b) | (c) |
|---|---|---|---|
| c→d | **23** (0w, 1w, 2w, 6, 7, 8w, 9w, 10, 11w, 12w, 13, 14, 15w, 16w, 19, 21w, 24w, 25, 26, 27, 29, 30w, 31) | 1 (3) | 8 (4, 5, 17, 18, 20, 22, 23, 28) |
| d→c | **13** (4w, 6, 12w, 13, 18, 19w, 21, 23, 24w, 26, 27, 30, 31w) | 8 (0, 1, 2, 9, 11, 14, 20, 22) | 11 (3, 5, 7, 8, 10, 15, 16, 17, 25, 28, 29) |

- Column by column, the re-pick row is very close to the iter04 row. That is expected:
  - c→d uses the same decoder and a neighbouring encoder;
  - in d→c, both models are neighbours of the iter04 picks.
- In d→c the re-pick turns a few iter04 failures into cats: #12 (lying tabby), #19 (tabby face) and #24 (striped
  kitten).
- #29 is borderline: a cat in roughly the same pose, but a doorframe appears in the background. I put it in (c).
  If it were accepted, d→c would reach 14/32.
- Compared with the iteration-05 validator's counts on the same columns (iter04: c→d 21, d→c 11), the re-pick is
  about +1-2 per direction. That is within the spread between counters.

Striking re-pick pairs:
- c→d: #6 (golden pup), #7 (white poodle), #13 (Boston terrier), #19 (fluffy white dog from an orange cat), #26 (dog
  among yellow flowers), #31 (Japanese chin).
- d→c (from `best_pairs_dog_to_cat_repick_big.png`): #307 (Pekingese → orange cat), #331 (→ grey tabby), #143 (→ tabby
  on grass), #26 (black dog → black cat), #53.

## Success criteria (validator)

| direction | KID | cost clamped | eye | met? |
|---|---|---|---|---|
| c→d (KID ≤ 7, cost ≤ 125, eye ≥ 23/32) | 8.34 ✗ | 123.2 ✓ | 23 ✓ | **no** (KID) |
| d→c (KID ≤ 5, cost ≤ 140, eye ≥ 14/32) | 4.66 ✓ | 136.8 ✓ | 13 ✗ (14 with #29) | **no on my strict count; borderline** |

The re-pick does not replace FT on any metric except judge rate. FT is still about 20-30 cheaper and 2-6 KID points
better, in both directions.

## Figures (4× NEAREST)

- **FT:** `ft_cat_to_dog_512_big.png`, `ft_cat_to_dog_512_b_big.png`, `ft_dog_to_cat_512_big.png`,
  `ft_dog_to_cat_512_b_big.png`, `ft_*_1000_lowest_ptarget_big.png`, `ft_best_pairs_*_1000_big.png`.
- **Re-pick:**
  - stage-1 probes: `probe_stage1_{cat_to_dog,dog_to_cat}_big.png` (every encoder, ranked, probe images 0-15);
  - outputs: `{cat_to_dog,dog_to_cat}_repick_big.png` and `_repick_b_big.png`;
  - lowest target probability: `*_repick_lowest_ptarget_big.png`;
  - best pairs: `best_pairs_*_repick_big.png`;
  - side by side: `compare_{cat_to_dog,dog_to_cat}_big.png` (input / FT / iter04 / re-pick, test images 0-31).
- Not redrawn: flow and steering grids, because no model was trained or changed.

## Problems and deviations

- **Smoke 1 failed** on an empty `_b` grid at n = 8. Its partial smoke outputs are in `/vol/iter06/smoke/`. After the
  fix, smoke 2 ran clean (`/vol/iter06/smoke2/`; its downloads are in `smoke/`).
- **Selection caveat:** the rule optimises a 64-image judge rate, which moves in steps of 1.6 points with a se of
  about 5. See the d→c caveat above.
- **Validity caveat:** the encoded latents are not N(0, I), which remains true for every pair. KID on T(x) is the
  check of the target marginal.

## Volume outputs (all new, under `/vol/iter06/`)

- `transport_{dir}_FT_512_999.pt` (inputs, latents, outputs), `ft1000.json`, `ft1000_per_image.json`.
- `probe_{dir}.pt` (64 inputs, latents of all 11 encoders, and all 41 probe outputs).
- `transport_{dir}_repick.pt` (512 images, with latents), `samples_{dir}_{FT,iter04,repick}.pt`.
- `evalT_stats.json`, `evalT_small.pt`, `memo_T_{cat,dog}.json`, `repick06.json`, and the figures.
