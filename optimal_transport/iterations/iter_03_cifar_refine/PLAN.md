# Iteration 03 plan: CIFAR-only refinement of the target models

Implements the spec in `../iter_02_purity/VALIDATION.md` §4 exactly.

**Why.** Iteration 02 showed that most of the purity gap sits in the target models' own sampling law (prior
samples: dog model 79.5% dog, cat model 80.1% cat; ceilings 87.7 / 85.1), with leftover other-CIFAR-class modes
from the all-class initialisation. Only a better model of ν improves a *valid* coupling. Continue fine-tuning
each model on CIFAR-10 train images of its own class only, with a large batch, then decode the latents already
saved in iteration 02 (encoded by the unchanged source models): T = decode_new ∘ encode_old.

**Code.** `optimal_transport/tasks_iter03.py` (imports iteration-02 tasks; adds `finetune_cifar`,
`decode_saved`, `memo_check`, `chain`) and `optimal_transport/modal_iter03.py` (same image/volume pattern,
ships `recovered_src/` fallbacks).

**Training** (`finetune_cifar`): init `out/{cls}.pth` (EMA weights, read-only), fresh Adam, CIFAR-10 *train*
images of the class only (5,000; never test), random hflip, bf16 autocast, grad-clip 1.0, bs 256 (OOM ->
gradient accumulation), lr warmup 200 steps to 1e-4, constant to 50% of the time cap, linear decay to 1e-5 at
the cap, EMA 0.999, cap 40 min or `GPU deadline (16:05 PDT) − 17 min`, whichever is smaller. Saves slim EMA
weights `/vol/iter03/{cls}_c3_mid.pth` (half cap) and `/vol/iter03/{cls}_c3.pth`.

**Container D (dog):** finetune → `prior_samples` (512, seed-0 z as in iteration 02) → decode
`/vol/iter02/latents_cat_to_dog.pt` → `/vol/iter03/transport_cat_to_dog_iter03_c3.pt` → (mid prior if > 20 min
left) → memorisation guard → `eval_runs` (judge, cost, clamped cost, KID vs all test target) → `evalD`.
**Container C (cat):** mirror (dog→cat, `evalC`).

**Memorisation guard:** median over samples of min-over-5,000-train squared L2, for new prior, old prior
(`/vol/iter02/prior_{cls}.pt`) and first 512 real test images; flag if new median < 0.5 × test median.

**Smoke first:** 20 steps at bs 256 + 10-step prior/decode on 4 images + memo + eval (one container).

**Success criteria (per direction):** judge target rate ≥ 79% and ≤ ceiling + 2; KID ≤ baseline (2.92 c→d,
2.16 d→c) within 1 se; clamped cost ≤ 110; prior samples improve (judge up vs 79.5 / 80.1, KID ≤ 3.28 / 4.42)
with no memorisation flag; by eye at least as many same-scene outputs as the baseline.

No new datasets are used in this iteration.
