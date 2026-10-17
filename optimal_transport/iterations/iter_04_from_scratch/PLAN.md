# Iteration 04 plan: cat-only and dog-only models trained from scratch

**Why.** The fine-tuned single-class models keep emitting other CIFAR classes (boat, frog, truck, horse,
deer) inherited from the all-class pretrained init. That impurity caps the judge rate of the composed ODE
at about 75% (ceilings 87.7% dog / 85.1% cat). Models trained from random init on one class never learn the
other classes. The open question is whether two **independently** trained models still map the same noise
to the same scene, which is what makes T = decode_dog ∘ encode_cat a low-cost coupling.

**Code.**
- `optimal_transport/tasks_iter04.py`: `train`, `snap_eval`, `pick_snapshot`, `chain_train`, `eval_composed`,
  `labeled_grid`. Reuses `tasks_iter02` (`prior_z`, `kid`, `_Incep`, `eval_runs`), `tasks_iter03.memo_check`,
  `evaluate.py` (judge, `grid_png`), `pf_ode.py`.
- `optimal_transport/modal_iter04.py`: Modal entry (chains `smoke_train`, `smoke_eval`, `train_cat`,
  `train_dog`, `eval`), one A10G per call.
- `common/score_sde_backend.py`: new functions only (`apply_model_overrides`, `build_network`,
  `load_model_auto`, `load_score_model_auto`). Existing `load_model` / `load_score_model` are unchanged.
  Every iter-04 checkpoint stores `config_overrides` and `config_name`; `load_model_auto` rebuilds the
  architecture from them (old checkpoints without the key load exactly as before).

**Architecture.** score_sde_pytorch DDPM++ continuous VP (`vp/cifar10_ddpmpp_continuous`) with overrides
nf=64, ch_mult=(1,2,2,2), num_res_blocks=2, attn_resolutions=(16,), dropout=0.1. **9,864,771 parameters**
(full model: 61,804,419). Random init (score_sde default init); no pretrained or fine-tuned weights loaded.
Same VP SDE (beta 0.1 to 20) as before.

**Training** (one A10G container per class, run in parallel: cat = 3, dog = 5). CIFAR-10 TRAIN images of the
class only (5,000), random horizontal flip, standard VP continuous objective (noise to s ~ U(eps, 1), regress
network output on the added noise, time label s·999, as in `separate_models/finetune.py`), batch 256, bf16
autocast, Adam lr 2e-4 with 1,000-step linear warmup then constant, grad clip 1.0, wall-clock cap 95 min
(checked every step).
- **EMA** (written deviation): decay_t = min(0.9995, (1+t)/(10+t)), the usual warm-started EMA. With a flat
  0.9995 the random init would still carry weight 0.9995^t (17% at t = 3,500), which would spoil the early
  snapshots that the memorisation analysis needs. From step ~18k on the decay is exactly 0.9995.
- **Snapshots** (EMA weights) at 12.5%, 25%, 50%, 75%, 100% of the 95-min cap:
  `/vol/iter04/{cls}_s{step}.pth` (5 snapshots; one extra early one, because memorisation grows with
  training on 5,000 images). Full training state `/vol/iter04/{cls}_last.pth` for resuming.

**Per-snapshot stats** (same container, after training): 256 prior samples from fixed seed-0 noise
(`prior_z(256)`, the first 256 of the iteration-02 noise), PF-ODE decode with **100 Heun steps** (eps 1e-3);
judge target-class rate (`/vol/judge/cifar10_judge.pth`, `evaluate.load_judge`); KID ×1e3 vs all 1,000 real
test images of the class (iteration-02 `kid`, Inception pool features), with a floor = 256 real train images
vs test; memorisation check `tasks_iter03.memo_check` (median nearest-train squared L2 of samples ÷ same for
the first 512 real test images). Output `/vol/iter04/snapstats_{cls}.json`.

**Snapshot choice rule.** Per class, among snapshots with memorisation ratio >= 0.5 (not flagged), take the
one with the lowest KID. If all are flagged, take the highest ratio and say so.

**Evaluation container** (one A10G, after both trainings): composed PF-ODE T = decode_dog ∘ encode_cat and
dog → cat on the first 512 test cats and dogs, 200 Heun steps, eps 1e-3. Standard harness
(`tasks_iter02.eval_runs`): cost unclamped and clamped (± se), judge target rate vs ceiling (87.7% dog /
85.1% cat), KID vs the train512 floor, lowest-p figure. Required figures (rendered in the container, 4×
NEAREST): `flow_grid_big.png`, `flow_grid_dog_to_cat_big.png` (8 images; raw ODE states at s ≈ 0.2..1.0
clamped to [-1,1]); `steering_grid_big.png` (8 cats shown; per-column mean cost and judge dog-rate computed on
the first **64** test cats, written to `steering_grid.json`; s* = 0.2..1.0 partial encodings, then
density-ratio guidance (1+w)s_dog − w s_cat with w = 0.25, 0.5, 1.0 at s* = 1; only s* = 1, w = 0 is a valid
coupling and is framed in green); `best_pairs_big.png` (rule: judge P(dog|output) >= 0.9 and
P(cat|input) >= 0.5, lowest cost first; also for dog → cat); `same_noise_both_models_big.png` (16 seed-0
noises decoded by the new cat and dog models, plus the fine-tuned models for reference) with a numeric
same-scene check: mean |cat(z) − dog(z)|² for matched noise vs mismatched noise (256 noises).

**Smoke first:** `smoke_train` (30 steps, 16 samples, 5 Heun steps) then `smoke_eval` (8 images, 10 steps).

**Budget.** ≤ 260 GPU-min: 2 × ~102 (train + snapshot eval) + ~15 eval + ~10 smoke ≈ 230. Max 2 containers.

**Compare against** the current best (fine-tuned models): cost 101.6 / 104.7, judge 74.8% / 74.6%,
KID×1e3 2.9 / 2.2, prior-sample rates 79.5% dog / 80.1% cat.

No new datasets.

## Recovery addendum (18:26 PDT)

Both training containers were stopped at about 18:15 PDT by a client-side cancellation ("Received a
cancellation signal while processing input"), at 66.9 (cat) and 65.2 (dog) training minutes, step ~17,000.
Not a timeout or budget limit. Only the snapshots before that exist: cat s3071 / s6140 / s12129 and dog
s3108 / s6195 / s12386 (12.5%, 25%, 50% of the cap); no `_last.pth` was written. GPU cost recorded
(68.5 min each).

1. `snapeval` (one container): per-snapshot stats on the six existing snapshots (same protocol as above).
   Result: KID still falls steeply with training (cat 136 → 30 → 15.3, dog 100 → 18 → 10.9 ×1e3 at n=256)
   and no snapshot is near memorisation (ratios 1.03 to 1.11), so the models are under-trained.
2. **Resume** (written deviation): `chain_resume` continues each class from its s12k EMA snapshot for **40 min**
   (fresh Adam, 200-step warmup to lr 2e-4, EMA 0.9995, step count continues), snapshots at 25/50/75/100% of
   the 40 min, per-snapshot stats on the new snapshots, then the pick is made over all snapshots of the class
   (same rule) in `snapstats_{cls}_all.json`. 40 min (not the 28 min left of the original 95) keeps the
   iteration under 260 GPU-min: 142 used + 2 × ~44 + ~12 eval ≈ 242.
3. Then the composed-ODE evaluation exactly as planned.

Launches now use the **deployed** app (`modal deploy optimal_transport/modal_iter04.py`, then
`optimal_transport/launch_iter04.py spawn <chain>`), so a job no longer depends on the local client;
the local fetcher runs under `caffeinate -is` and only downloads results.
