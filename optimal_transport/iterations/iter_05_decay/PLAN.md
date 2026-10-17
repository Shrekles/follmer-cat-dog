# Iteration 05 plan: cosine-lr resume of the from-scratch nets, mixed pairs, Tweedie step

Implements the spec in `iter_04_from_scratch/VALIDATION.md` §5 (authoritative). Training agent, 2026-10-16 ~20:30 PDT.

**Why.** The iteration-04 from-scratch nets (9.86M params) fixed the other-class modes but their KID plateaued at
14.1 (cat) and 8.2 (dog) after ~60 min at a constant lr 2e-4 with a short EMA (0.9995, a horizon of ~8 min). A
cosine lr decay plus a long EMA (0.9999) is the standard cure for a plateau caused by lr noise. It also separates
lr noise from capacity limits. Mixed encoder/decoder pairs test which leg drives cost and sharpness, and whether
fine-tuned latents decoded by a pure decoder combine low cost with purity.

**Code (new files only).**
- `optimal_transport/tasks_iter05.py`: `resume_decay`, `snap_eval05`, `pick05`, `chain_decay`, `snap_eval_existing`
  (recovery), `gauss_stats` / `gauss_flags`, `tweedie`, `eval05`, `chain_smoke`. Reuses iteration-02/03/04
  helpers (`eval_runs`, `kid`, `prior_z`, `memo_check`, `_segments`, `labeled_grid`, the auto loaders in
  `common/score_sde_backend.py`).
- `optimal_transport/modal_iter05.py` (app `follmer-cat-dog-iter05`, A10G, timeout 2.4 h) and `launch_iter05.py`
  (deployed app: `modal deploy` + `spawn`; `fetch` runs locally under `caffeinate -is` and only downloads small
  files).
- Outputs only under `/vol/iter05/` (smoke: `/vol/iter05/smoke/`). `/vol/iter04/`, `iter02/`, `iter01/` and
  `out/` are read only.

**Training (containers A = cat, B = dog, in parallel).** `resume_decay(cls)`:
- loads `iter04/{cls}_last.pth`: the raw weights, the Adam state (checked: it loads; Adam step 10,328 for cat) and the
  step (22,457 cat / 22,697 dog). Both EMA trackers start from the stored 0.9995 EMA.
- Data and loss as in iteration 04: CIFAR-10 train images of the class only, hflip, VP continuous loss, batch 256,
  bf16 autocast, grad clip 1.0, label s·999.
- lr follows a time-based cosine from 2e-4 to 2e-6 over 45 min (p = elapsed / cap, checked every step), with no
  warmup because the Adam state loads. EMA decays are fixed at 0.9995 and 0.9999. Seed 5.
- At 50/75/100% of the cap it saves `{cls}_e9995_s{step}.pth`, `{cls}_e9999_s{step}.pth` and the resumable state
  `{cls}_last_s{step}.pth` (raw weights, both EMAs, Adam), then runs `vol.commit()`. A cancelled run can continue
  from the latest `_last_s*` with `p0` set to its progress.

**Per-snapshot stats** (same container): the 50% 0.9999 EMA, both EMAs at 75% and at 100%, and the iteration-04
pick (cat_s17297 / dog_s20119) as a reference. Each uses the transport protocol: n = 512, `prior_z(512)` (seed 0),
200 Heun steps, eps 1e-3. Reported per snapshot:
- judge rate and KID ×1e3 against all 1,000 test images of the class, plain and with the final Tweedie step;
- the memo ratio;
- the fraction of pixels outside [−1, 1] and the saturated-sample fraction (more than 50% of pixels clipped).

**Pick rule (pick05):**
1. Discard snapshots with memo ratio < 0.5 (warn below 0.8).
2. Take the lowest KID.
3. Among snapshots within 1 se of that lowest KID (its own se), take the highest judge rate that is ≤ ceiling + 2 points.
4. If still tied, take the latest step, then the longer EMA.

The iteration-04 reference is reported but is not eligible.

**Eval container (after A and B)**, `eval05`:
1. Composed ODE with the new picks on the first 512 test cats and 512 test dogs, 200 Heun steps, eps 1e-3. Latents go
   to `/vol/iter05/latents_*.pt` and outputs to `transport_*_iter05.pt`. Each output set is reported plain and with a
   final Tweedie step x̂₀ = (x + σ²(eps)·score(x, eps)) / α(eps), using the decoder. This is one network evaluation
   and deterministic, so T stays a map.
2. Mixed pairs, decode only from saved latents:
   - M1: FT cat latents (`iter02`) → new dog decoder;
   - M2: FT dog latents → new cat decoder;
   - M3: new cat latents → FT dog decoder `out/dog.pth`;
   - M4: new dog latents → FT cat decoder `out/cat.pth`.
   Each is reported plain and with the Tweedie step.
3. The FT baseline (`iter01` outputs) and iteration 04 (`iter04` outputs) are re-scored in the same run, plain and with
   their own decoder's Tweedie step, so every row shares the judge, the KID references and the inputs.
4. Gaussianity of every latent set, against a fresh N(0, I) sample (seed 1) and against `prior_z` (seed 0). The sets
   are FT (`iter02`), iter04 and iter05, cat and dog encoders. Statistics:
   - mean, std, and the distribution of per-image ‖z‖²/d;
   - pooled energy ratio k²·E[avgpool_k(z)²] for k = 4 and 8 (iid N(0,1) gives 1), with a bootstrap se over images;
   - lag-1 autocorrelation, horizontal and vertical;
   - the KS statistic of the pixel values against N(0, 1).
   A set is flagged if a pooled ratio differs from the fresh N(0, I) value by more than 3 × its own bootstrap se.
5. Standard eval (`eval_runs`) on all of these: cost unclamped and clamped ± se, judge target against the ceiling,
   source rate, KID against the train-512 floor, lowest-p figures, overshoot, and the saturated fraction.
6. Required figures (4× NEAREST, rendered in the container):
   - `flow_grid_big.png` / `flow_grid_dog_to_cat_big.png`: Tweedie estimates x̂₀(x_s) at s = 0.1, 0.2, 0.35, 0.5,
     0.75, 1.0 on the source leg and at 0.75 … 0.1 on the target leg, plus a second row per image with x_s/α_s;
   - `steering_grid_big.png`: s* = 0.2 … 1.0, then w = 0.25, 0.5, 1 at s* = 1, with statistics on the first 64 cats.
     Only s* = 1 with w = 0 is a coupling;
   - `best_pairs_big.png` / `best_pairs_dog_to_cat_big.png`, using the strict rule p_target(out) ≥ 0.9 and
     p_source(out) ≤ 0.05, clamped cost ascending. The same rule applies to the mixed pairs and to FT
     (`best_pairs_<run>_big.png`);
   - `same_noise_both_models_big.png`: iter05, iter04 and FT pairs on seed-0 noise, with matched and mismatched costs;
   - `compare_{cat_to_dog,dog_to_cat}_big.png`: test images 0-31, with rows input / FT / iter04 / iter05 / mixed
     (FT encoder → new decoder) / mixed (new encoder → FT decoder). This is the figure for counting by eye.

**Budget.** The spec estimates 141 GPU-min: smoke 3, A 58, B 58, eval 22. The cap for this iteration is ~150 GPU-min.
Run 2 stood at 234.1 before this iteration. The smoke cost 2.2 GPU-min (total 236.3). At most 2 containers run at
once.

**Success criteria (from the spec).** All are judged on the valid coupling s* = 1, w = 0, for each direction and each pair.
1. Priors at n = 512 / 200 steps: KID ≤ 10 for cat and ≤ 6 for dog; judge between 82% and ceiling + 2; memo
   ≥ 0.8; saturated ≤ 0.5%.
2. Composed iter05 pair: KID ≤ 6 in both directions; judge ≥ 80% / 77%; clamped cost ≤ 132 / 140.
3. By eye on the comparison figures: c→d ≥ 21 / 32 same-scene with ≤ 6 broken; d→c ≥ 16 / 32 with ≤ 8 broken.
4. A pair replaces FT as best valid coupling only if KID ≤ FT KID + 1 se, judge ≥ FT + 2 points, and its eye count is at
   least FT's (59% / 56%).

No new datasets.

## Addendum (21:29 PDT, after the snapshot stats)

The snapshot stats showed KID getting worse with decay, and the iteration-04 picks best in both classes. I added
mixed pairs M5 and M6 (FT latents → iter-04 decoders, ~3 GPU-min) and redeployed. The eval spawn ran the previous
deployment, so M5 and M6 did not run. The code was reverted to what ran (see REPORT). The eval used the iter-05 picks
as specified.
