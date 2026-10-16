# Iteration 02 - purity of the target marginal (training agent plan)

Written 13:30 PDT 2026-10-15, before any GPU launch. Implements the spec in
`iterations/iter_01_judge_and_eval/VALIDATION.md` section 4, with the orchestrator's amendment.

## Why
Iteration 01: composed ODE cat->dog cost 101.6 +- 3.4, judge dog-rate 74.8% vs real-dog ceiling 87.7%;
~30% of columns look unchanged by eye. The bottleneck is purity of the second marginal, not cost.

## Jobs (Modal, A10G, at most 2 containers at once; code `optimal_transport/tasks_iter02.py`,
`optimal_transport/modal_iter02.py`; outputs in `/vol/iter02/`)
- **Smoke** (n=4, 10 ODE steps, 20 train steps on 512 images, CINIC replaced by an HTTP HEAD check): ~5 GPU-min.
- **Container A** (~55 GPU-min): (1) prior samples: the same 512 z (seed 0) decoded by cat and dog models;
  (2) encode + save latents of first 512 test cats (cat model) -> `latents_cat_to_dog.pt`;
  (3) guided decodes cat->dog with score (1+w) s_dog - w s_cat, w = 0.5, 1.5; (4) latents of first 512 test dogs;
  (5) dog->cat w = 1.0; (6) eval (judge, cost unclamped+clamped, KID) incl. the reference KID rows and the
  iteration-01 baseline files `/vol/iter01/transport_{cat_to_dog_B,dog_to_cat_A}.pt`.
- **Container B** (~60 GPU-min): download CINIC-10 inside Modal to `/vol/data/cinic10/` (12-min cap), keep only
  ImageNet-derived files (names not starting with `cifar10-`) of class dog (and cat, stored for later);
  fine-tune `out/dog.pth`: batch 128 (64 CIFAR-train dogs + 64 CINIC-ImageNet dogs), lr 1e-4 with 200 warm-up
  steps, EMA 0.999, bf16 autocast, grad clip 1, random h-flip, 32-min training cap (shortened if the
  14:55 GPU deadline requires) -> `/vol/iter02/dog_cinic.pth` (+ `dog_cinic_mid.pth`); then prior samples
  with the same z and cat->dog = decode A's cat latents with the new dog model (w = 0); eval.
  Fallback if the download fails: guided cat->dog w = 1.0 and dog->cat w = 0.5 from A's latents.

**Dataset source:** CINIC-10, official Edinburgh DataShare:
https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz (Darlow et al. 2018).
CIFAR-10 test images are excluded by construction (all `cifar10-*` files are dropped).

## Validity (orchestrator amendment)
Guided decoding with (1+w) s_tgt - w s_src samples a tilted law ~ p_dog^(1+w) / p_cat^w, not nu, so (like
s* < 1) it is NOT a coupling of mu and nu. Guided rows are diagnostics only. The only candidate for a new best
valid coupling is the composed ODE with the CINIC-fine-tuned dog model (w = 0).

## Evaluation
First 512 test cats / dogs. Cost E|x - T(x)|^2 on unclamped and clamped outputs (+- se); judge target-rate vs
ceiling; KID x1e3 (Inception-v3 IMAGENET1K_V1 pool features, 299 bilinear, cubic polynomial kernel, unbiased
MMD^2, bootstrap se) vs all 1000 real test images of the target class, with reference rows: 512 real train
dogs (floor), 512 real test cats (wrong class), iteration-01 baseline. Figures: 16-column input/output for every
run, prior samples, and the 16 lowest-p_target outputs per run.

## Budget / time
Smoke ~5 + A ~55 + B ~60 = ~120 GPU-min (session ~165 of 480). All GPU work must end by 14:55 PDT
(tasks check an absolute deadline and skip what cannot finish); deliverables by 15:05.

## Note on the local disk
The Mac disk is full (~360 MB free) and iCloud has evicted many repo files (`evaluate.py`, `pf_ode.py`,
`AGENT_CONTEXT.md`, `VALIDATION.md`, ...): they cannot be re-downloaded ("No space left on device"). I
reconstructed them byte-exactly (sizes match) from the session transcripts; copies used for Modal are in
`recovered_src/`. Nothing was deleted.
