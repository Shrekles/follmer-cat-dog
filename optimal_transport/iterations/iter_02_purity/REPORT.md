# Iteration 02 report: purity of the target marginal

Training agent, 14:45 PDT 2026-10-15. Spec: `iter_01_judge_and_eval/VALIDATION.md` section 4, plus the
orchestrator's amendment (guided decoding is not a valid coupling). Code: `optimal_transport/tasks_iter02.py`,
`optimal_transport/modal_iter02.py`, local post-processing `build_outputs.py` (this folder). Volume outputs in
`/vol/iter02/` (latents `latents_{cat_to_dog,dog_to_cat}.pt` saved for reuse; `dog_cinic.pth`,
`dog_cinic_mid.pth`; prior and transport files). CINIC data: `/vol/data/cinic10/CINIC-10.tar.gz`,
`/vol/data/cinic10_{dog,cat}_imagenet.pt`.

## Results (first 512 test cats / dogs; n = 512 per row; judge-rate se ~ 0.015-0.02)

Judge ceilings: real test dogs 87.7% (first 512: 88.5%), real test cats 85.1% (85.4%).
KID x1e3 against all 1000 real test images of the target class. Reference rows: **512 real train dogs vs test
dogs 0.36** (floor), 512 real test cats vs test dogs 30.6 (wrong class); train cats vs test cats 0.09, test dogs
vs test cats 29.7.

| run | valid coupling? | cost (unclamped) | cost (clamped) | judge target-rate (ceiling) | still source | KID x1e3 |
|---|---|---|---|---|---|---|
| cat->dog baseline (iter 01, 16k dog model) | yes | 101.6 +- 3.4 | 101.6 +- 3.4 | 74.8% (87.7%) | 16.2% | 2.92 +- 0.92 |
| **cat->dog, CINIC-fine-tuned dog model, w = 0** | yes (up to model error) | **95.7 +- 3.3** | **95.4 +- 3.3** | **47.3%** (87.7%) | 21.9% | **14.26 +- 1.76** |
| cat->dog guided w = 0.5 | **no (diagnostic)** | 187.4 +- 5.3 | 187.3 +- 5.3 | 89.5% | 3.9% | 1.89 +- 0.50 |
| cat->dog guided w = 1.5 | **no (diagnostic)** | 414.4 +- 9.2 | 413.7 +- 9.2 | 96.1% | 0.2% | 12.69 +- 1.52 |
| dog->cat baseline (iter 01) | yes | 104.7 +- 3.4 | 104.6 +- 3.4 | 74.6% (85.1%) | 12.9% | 2.16 +- 0.64 |
| dog->cat guided w = 1.0 | **no (diagnostic)** | 281.7 +- 7.4 | 281.3 +- 7.4 | 87.7% | 0.6% | 4.87 +- 0.75 |

Guided rows use the score (1+w) s_tgt - w s_src. That decodes towards a tilted law ~ p_tgt^(1+w) / p_src^w,
not nu, so (like s* < 1 in iteration 01) these rows are **not couplings of mu and nu and do not bound W2^2**.

**Prior samples** (the same 512 noise vectors, seed 0, decoded by each model; Heun 200):

| model | judge dog | judge cat | KID x1e3 vs its class | T(source) target-rate for comparison |
|---|---|---|---|---|
| cat model (`out/cat.pth`) | 7.2% | **80.1%** | 4.42 +- 0.70 | T(dogs) cat-rate 74.6% |
| dog model (`out/dog.pth`) | **79.5%** | 11.1% | 3.28 +- 0.77 | T(cats) dog-rate 74.8% |
| CINIC dog model (`dog_cinic.pth`) | **56.1%** | 18.6% | 8.08 +- 1.21 | T(cats) dog-rate 47.3% |

## Reading
1. **Prior-sample diagnostic: impurity is mostly in the models themselves.** The dog model's own samples
   are judged dog only 79.5% of the time (ceiling 87.7%), and 11% are judged cat. T(cats) is a bit lower
   (74.8%, gap 4.7 points, about 2 se) and the cat model shows the same pattern (80.1% vs 74.6%). So most of the
   ~13-point gap to the ceiling is the target model's sampling law, with a smaller extra loss from the
   encode leg (latent mismatch).
2. **The CINIC fine-tune made the dog model worse for nu = CIFAR dogs.** 4,718 steps x 128 (604k images,
   32 min, loss flat at ~0.031-0.033). The judge's dog-rate fell from 79.5% to 56.1% on prior samples and from
   74.8% to 47.3% on T(cats). KID to real CIFAR test dogs went from 3.3 to 8.1 (prior) and 2.9 to 14.3 (T(cats)).
   Judge histograms for T(cats) with this model: dog 242, cat 112, horse 39, bird 33, airplane 27, versus
   baseline dog 383, cat 83. The cost fell a little (95.7 vs 101.6), but that comes with a worse second marginal,
   so it is **not** a better valid coupling. Likely cause: ImageNet-derived 32x32 images differ in style from
   CIFAR (other framing, more downsampling, other colours), and a 50/50 mix moved the model towards that style.
   Both the CIFAR-trained judge and the KID against CIFAR test dogs penalise exactly that. By eye
   (`cat_to_dog_compare_big.png`, row 3) the CINIC outputs look much like the baseline: the same columns
   stay cat (0, 3, 15), and a few are messier (5, 14). Only 32 min of training, and no lower CINIC share was
   tried.
3. **Guidance raises the judge rate but breaks the "same scene" property.** w = 0.5 gives 89.5% (about at
   the real ceiling, under the 90% overshoot limit) and the best KID of any run (1.89; baseline 2.92, but the
   difference is about 1 se). However, the cost almost doubles (187). w = 1.5 overshoots (96.1% > ceiling: a
   sharpened, mode-seeking law), KID gets much worse (12.7) and cost is 414: the outputs are new
   portrait-style dogs (row 5 of the compare figure). dog->cat w = 1.0: 87.7% cat, KID worse than baseline
   (4.9 vs 2.2), cost 282. In the compare figure, w = 0.5 turns several unchanged columns into dogs (0, 5, 6, 14)
   but often replaces the scene with a centred close-up dog. Whatever it does, it is not a valid coupling.
4. Clamping to [-1, 1] changes costs by < 0.7 everywhere (at most 0.6 for w = 1.5).

**Against the spec's success criterion** (judge >= 82% and <= 90%, KID better than baseline, clamped cost <= 110,
visual): no run passes. CINIC w = 0 fails on judge (47%) and KID (14.3). Guided w = 0.5 passes judge and KID but
fails cost (187) and is not a valid coupling anyway. **The best valid coupling is still the iteration-01
baseline: cost 101.6 / 104.7, judge 74.8% / 74.6%, KID 2.9 / 2.2.**

## Figures (x4 NEAREST)
- `<run>_big.png`: rows input / output, test images 0-15. `<run>_b_big.png`: images 16-31.
  `<run>_lowest_ptarget_big.png`: the 16 inputs/outputs with the LOWEST judge p(target).
- `cat_to_dog_compare_big.png`: rows input / baseline / CINIC w=0 / guided w=0.5 / guided w=1.5 (same 16 cats).
  `dog_to_cat_compare_big.png`: rows input / baseline / guided w=1.0.
- `prior_same_z_big.png`: rows cat model / dog model / CINIC dog model, decoded from the same z.
  `prior_<model>_big.png`: 32 prior samples. `prior_<model>_lowest_ptarget_big.png`: the 16 least target-like.
- Data: `metrics.json` (summary table, KID refs, ceilings, full per-run stats), `evalA_*`, `evalB_*`
  (stats JSON + small tensors with cloned slices only).

## GPU usage (A10G, container minutes)
| job | minutes |
|---|---|
| smoke (first attempt failed locally at import, no container) | ~0 |
| smoke | 1.0 |
| A, first launch (prior samples done, then **cancelled at 13:45:49**) | ~11.5 (estimated from the log) |
| B, first launch (CINIC download + build done in ~9.5 min, then **cancelled at 13:45:56**) | ~10.5 (estimated) |
| A2 (latents, 3 guided decodes, eval) | 39.4 |
| B2 (CINIC reused, fine-tune 32 min, prior, decode, eval) | 42.3 |
| **total iteration 02** | **~105** (session ~150 of 480) |

## Problems / caveats
- **Both first containers were cancelled at the same moment (13:45:49 / 13:45:56). Modal's only message was
  "Function call was cancelled by user or a failure".** No agent transcript in this session contains a stop
  command. It was not reported as a spending limit, so I relaunched once (A2 without the finished prior
  step; B reused the CINIC tensors already on the volume). Cause unknown; ~22 GPU-min were partly lost.
- **Local disk is full (~270 MB free) and iCloud has evicted many repo files** (dataless): `AGENT_CONTEXT.md`,
  `VALIDATION.md`, `evaluate.py`, `pf_ode.py`, `modal_iter.py`, iteration-01 figures, logs and others. Reading them
  timed out ("No space left on device" when iCloud tried to download). I rebuilt the text files byte-exactly
  (sizes match) from the session transcripts. Copies are in `recovered_src/` and `modal_iter02.py` ships
  them when the originals are evicted. Iteration-01's `modal_iter.py` uploads the whole `optimal_transport/`
  folder, so it will hang/fail while files are evicted. Freeing disk space would fix this; I deleted nothing.
- The smoke test's HTTP HEAD on the CINIC URL returned a 4 KB HTML page. A GET follows redirects to the 687.5 MB
  gzip (checked with a range request), and the real download plus build took ~9.5 min: 21,000 ImageNet dogs
  and 21,000 ImageNet cats, with 6,000 `cifar10-*` files of each class excluded.
- `tasks_iter.py`: added a `folder="iter01"` parameter to `transport_pf` / `judge_eval` (the default keeps
  iteration-01 behaviour) and `.clone()` on the saved `x` slice.
- KID se is a bootstrap over both sets (duplicate-index pairs excluded). For n = 512 the se is large compared
  with the differences among good runs (~0.5-0.9 x1e-3).
- The judge over-credits outputs (validator, iter 01). The visual checks here are my quick look at the compare
  figure, not a full count.
- ImageNet-derived CINIC images differ in style from CIFAR's (see Reading 2). That is the main suspected reason
  the fine-tune hurt.

## Suggestions for iteration 03 (not run)
CINIC as used here is harmful. Options: a much smaller CINIC share (e.g. 10-20%) or CINIC only as pre-training
followed by CIFAR-only annealing, or the same for the cat model. Since the impurity is mostly the model's own
sampling law, a CIFAR-only dog model fine-tuned longer (or from a better init) is the more direct lever. A small
guidance weight (w = 0.1-0.25) is only a diagnostic.
