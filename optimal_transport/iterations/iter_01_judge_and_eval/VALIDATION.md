# Validation of iteration 01 (judge + evaluation harness + baseline)

Validator, 13:20 PDT 2026-10-15. No GPU used. Local work: viewed every `*_big.png` (also split into
8-column crops and 8x crops of disputed columns), loaded `small.pt` and recomputed costs, read
`metrics.json`, `REPORT.md`, `PLAN.md`, `tasks_iter.py`, `modal_iter.py`, `evaluate.py`.

## 1. Visual counts (full encode, s* = 1, the only valid coupling)

Categories: (a) convincing target animal AND same scene/pose/colors; (b) still looks like the source
animal (barely changed); (c) broken/blurry, or a target animal in a clearly different scene.
Judgments at 32x32 are uncertain for perhaps 2-3 columns per row of 16.

| figure | (a) same-scene translation | (b) unchanged source | (c) broken / other | judge target-calls, same 16 |
|---|---|---|---|---|
| cat->dog, test 0-15 (`cat_to_dog_full_big`) | 9 | 5 (0 grey cat, 3 yawning, 4 orange, 9 orange-in-grass, 15 lying on plaid) | 2 (5 scene replaced by green/white blob; 13 tiny unclear figure, judge says bird) | 14 |
| cat->dog, test 16-31 (`_full_b_big`) | 10 | 4 (19, 21, 24 siamese, 28) | 2 (17 and 23: good dog, but new composition) | 10 |
| dog->cat, test 0-15 (`dog_to_cat_full_big`) | 9 | 6 (2 yorkie, 6, 9, 11, 13, 15) | 1 (7 black dog + orange smear) | 13 |
| dog->cat, test 16-31 (`_full_b_big`) | 9 | 4 (18, 21, 22, 31) | 3 (16, 17, 20 distorted/abstract) | 15 |
| **total** | **37/64 (58%)** | **19/64 (30%)** | **8/64 (12%)** | **52/64 (81%)** |

Where translation works it is very good (e.g. cat->dog 2, 6, 7, 8, 11, 22, 27; dog->cat 4, 12, 24, 26,
27, 30: same background, pose and palette, clearly the other animal). This is the qualitative result
we want; the problem is the 30% of columns that do not change and the ~12% broken ones.

**Judge vs eye disagreement (important).** cat->dog: judge 24/32 dog vs ~19/32 convincing by eye
(plus 3 dogs in a new scene). dog->cat: judge 28/32 cat vs ~18/32 by eye. At 8x zoom, dog->cat
columns 2, 6, 15 are visually the unchanged input dog, yet the judge gives p(cat) = 0.99, 1.00, 1.00
(column 9: 0.60). So the ResNet judge is **over-crediting** some outputs, most likely reacting to
generator texture (cat-model "fur statistics") rather than shape. The reported 74.8% / 74.6% target
rates should be read as optimistic upper bounds; a distribution-level check that does not use the
same classifier is needed (spec below adds KID with Inception features).

**s* sweep figures.** Class identity flips mostly between s* = 0.4 and 0.6; rows s* = 0.6, 0.8, 1.0
are visually almost identical (a few late flips, e.g. cat->dog col 8 Pekingese appears only at
0.8, col 14 spaniel at 0.6). Crucially, the columns that stay "unchanged" at s* = 1 are unchanged at
every s* - they are not a case of "not enough noise". With alpha(1) = 0.007 the latent is almost
pure noise, so T(cats) is distributed (approximately) like the dog model's own samples. The
unchanged outputs therefore indicate that **the dog model (and cat model) themselves put mass on
other-class-looking images**, or that the encoded latents of those cats sit in a cat-favouring
region of latent space. Iteration 01 did not measure which; the spec below does (prior-sample
purity). Note cost keeps rising 62 -> 90 -> 102 from s* = 0.6 to 1 with little visible change: the
extra cost is fine detail/texture, not class.

## 2. Number checks

- Per-image costs in `small.pt` recompute exactly from stored x and outputs (first 64, all four s*,
  both directions). Full-encode means 101.6 +- 3.4 / 104.7 +- 3.4 reproduce iteration 00 exactly.
- Inputs in `small.pt` are in [-1, 1]; outputs range about [-1.08, 1.08], 0.5-0.9% of pixels
  outside. Cost is computed on unclamped outputs, judge on clamped ones. Clamping only lowers the
  cost (x lies in the cube), so the reported cost is a slight overestimate; harmless, but report
  cost on clamped outputs too from now on (real images live in [-1, 1]).
- Judge ceilings are consistent: 85.1% / 87.7% on all 1000 test cats/dogs, 85.4% / 88.5% on the
  first 512 used as inputs. Comparing output rates with these ceilings is fair.
- Leakage: the judge trains on `cifar10_train.pt` only; `separate_models/finetune.py` uses
  `cifar10_train.pt` only; evaluation uses the test split. No leakage found.
- Note that the first 64 cats have lower mean cost (87.0) than all 512 (101.6) - figures show a
  slightly easier-than-average subset; no bias in the numbers.
- `tasks_iter.transport_pf` hard-codes `/vol/iter01/`; new tasks need their own folder. Latents are
  not saved, so every new decoding experiment currently has to re-encode (5 GPU-min per 512).

## 3. Verdict on the current best (composed ODE, 16k-step models)

- cat->dog: cost **101.6 +- 3.4**, judge dog-rate **74.8%** vs real-dog ceiling 87.7% (gap 12.9
  points), 16.2% still called cat. By eye ~58% are convincing same-scene translations.
- dog->cat: cost **104.7 +- 3.4**, judge cat-rate **74.6%** vs ceiling 85.1% (gap 10.5), and by eye
  noticeably worse than the judge says.

This is a good-looking but **not yet valid** coupling of mu and nu: its second marginal is the dog
model's sampling law, which (by judge and by eye) contains too many cat-like and broken images. The
cost 102 is therefore not yet an honest upper bound on W2^2(cats, dogs). The bottleneck is
**marginal purity, not transport cost**. Lowering cost (reflow, better ODE) before the second
marginal is right would not be progress. The partial-encode sweep (s* < 1) is a diagnostic only.

Against option (ii) reflow now: reflow preserves the marginals of the coupling it starts from - i.e.
it would preserve the impure dog-model marginal - and needs many (x, T(x)) pairs (5,000 train cats =
~100 GPU-min of ODE just to build pairs at 200 Heun steps) plus long U-Net training; the earlier
5,000-pair Brenier flow came out blurry. Not affordable or useful this session until purity is fixed.
Option (iii) (ODE settings) has little room: encode/decode round trip error is already 0.003.

## 4. SPEC for iteration 02: `iter_02_purity`

**Hypothesis.** The ~16% unchanged / ~12% broken outputs come from the target model's own
distribution (impurity inherited from the all-class init and from only 5k images per class). Fixing
the target marginal (H1: density-ratio guidance, no training; H2: more dog data via CINIC-10) raises
the judge target-rate toward - not above - the real-image ceiling, with cost within ~10% of 102.
A prior-sample diagnostic tells which mechanism is responsible.

### Container A (A10G) - diagnostic + guidance, no training (~55 GPU-min)
1. **Prior-sample purity.** 512 noise vectors z ~ N(0, I) (seed 0); decode the SAME z with the
   cat model and with the dog model (Heun 200, eps 1e-3, batch 256-512). Save to
   `/vol/iter02/prior_{cat,dog}.pt`. Judge rates for these tell whether T(cats)'s dog-rate (74.8%)
   equals the dog model's own dog-rate (=> model impurity) or is lower (=> latent mismatch).
2. **Save latents.** Encode the first 512 test cats with the cat model (s: eps -> 1) and save
   `/vol/iter02/latents_cat_to_dog.pt` (x, z). Do the same for the first 512 test dogs with the dog
   model -> `latents_dog_to_cat.pt`. Reuse these in all later decode experiments (saves 5 min each).
3. **Density-ratio guided decode.** Decode with score
   `s_g(x, s) = (1 + w) * s_tgt(x, s) - w * s_src(x, s)` (s_tgt - s_src = grad log p_tgt/p_src at
   noise level s, an implicit cat-vs-dog classifier from our own two models; no external classifier).
   Encode leg unchanged. Runs: cat->dog w = 0.5 and w = 1.5; dog->cat w = 1.0. Each guided decode is
   2 network evals per step (~10 GPU-min per 512). Save as `/vol/iter02/transport_<dir>_w<w>.pt` in
   the iteration-01 format (`x`, `outputs`, `meta`, with `valid_coupling` explained in meta).
   Order: 1, 2 (cat->dog), 3 (cat->dog w 0.5, 1.5), 2 (dog->cat), 3 (dog->cat w 1.0).

### Container B (A10G; A100 only if bf16 batch 128 does not fit) - CINIC-10 dog model (~60 GPU-min)
1. Download CINIC-10 inside Modal to `/vol/data/cinic10/` from the official Edinburgh DataShare
   URL `https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz` (~690 MB; record in
   PLAN.md). **Cap 12 min** for download + extract + build; if exceeded, abort B (fallback below).
2. Build `/vol/data/cinic10_dog_imagenet.pt` (and `_cat_` for a later iteration): uint8 (N,3,32,32)
   from all three CINIC splits, **only ImageNet-derived files (names not starting with `cifar10-`)**.
   This excludes every CIFAR-10 test image by construction (and CIFAR train duplicates; we add our
   own `cifar10_train.pt` dogs instead). Expect ~21,000 dogs. Log counts.
3. Fine-tune the dog model: init from `/vol/out/dog.pth` (EMA weights of the 16k model), fresh Adam,
   **batch 128**, lr 1e-4 with 200-step warm-up then constant, EMA 0.999, bf16 autocast, the same VP
   loss as `separate_models/finetune.py`. **Each batch is 50% CIFAR-train dogs, 50% CINIC-ImageNet
   dogs** (keeps the model anchored to CIFAR style, which is what nu is), random h-flip. **Wall cap
   32 min of training** (expect ~6-7k steps, i.e. ~0.85M images). Save EMA to
   `/vol/iter02/dog_cinic.pth` (+ snapshot at the midpoint `dog_cinic_mid.pth`). Never overwrite
   `/vol/out/*`.
4. Evaluate in the same container: 512 prior samples from the new dog model with the SAME z as A1
   (seed 0); cat->dog transport = encode first 512 test cats with the old cat model, decode with
   `dog_cinic.pth` (unguided). If A's `latents_cat_to_dog.pt` already exists on the volume
   (`vol.reload()`), decode from it instead of re-encoding.

### Eval container (A10G, ~5 GPU-min), after A and B
`judge_eval`-style stats for every new transport file (cost on unclamped AND clamped outputs, +- se;
judge target-rate, source-rate, histogram) and for the prior-sample files (target-class rate).
**Add KID** (Inception-v3, torchvision `IMAGENET1K_V1` weights downloaded in Modal, inputs resized to
299 bilinear, 2048-d pool features, polynomial kernel, unbiased MMD^2, report x1e3 with bootstrap
se) of each output set against **all 1000 real test dogs** (or cats). References in the same table:
KID(512 real train dogs, test dogs) = floor, KID(512 real test cats, test dogs) = wrong-class level,
KID(iteration-01 baseline T(cats), test dogs) from `/vol/iter01/transport_cat_to_dog_B.pt`. Download
stats JSON + small tensors (save `.clone()`d slices, < 30 MB). Figures: the standard 16-column
input/output figure for every run, plus one figure of the 16 outputs with the LOWEST p_target, so
failures are visible rather than averaged away.

### Success criterion (cat->dog headline; dog->cat for w-guidance only)
A run counts as progress only if ALL hold:
- judge dog-rate on T(cats) **>= 82%** (gap to the 87.7% ceiling at most ~6 points, from 12.9) and
  **not above ~90%** (exceeding the real-data ceiling means a sharpened, mode-seeking law - also not nu);
- KID to real test dogs **lower than the baseline's** and within 2 se of the real-train-dog floor
  or clearly closer to it than baseline;
- cost (clamped) **<= 110** (<= +8% over 101.6);
- visual: in columns 0-15, at most 2 still-cat outputs and at least 11 same-scene dogs.
Report the prior-sample dog-rate of each dog model next to its T(cats) rate.

### Budget / time
A ~55 GPU-min, B ~60 GPU-min (12 data + 32 train + ~15 eval), eval ~5, smoke ~4: **~125 GPU-min**
(session total ~170 of 480). Run a smoke (n=4, 10 ODE steps, 20 train steps on a 512-image subset,
CINIC step replaced by a 1-file existence check) first; then A and B in parallel. Expected wall time
~70 min from launch (launch by ~13:45 -> results ~15:00), leaving one more ~60-min iteration before
the 16:05 last-start.

### Fallbacks
- CINIC download slow/fails (cap hit): B instead runs guided cat->dog w = 1.0 and dog->cat w = 0.5
  from A's saved latents (wait for `vol.reload()` to show them) - pure guidance iteration.
- Guidance raises judge rate but KID gets worse or judge rate overshoots ceiling: guidance is
  distorting the marginal; keep the unguided coupling, use smaller w (0.25) in iteration 03.
- Prior-sample dog-rate is clearly HIGHER than T(cats) dog-rate (latent mismatch, not impurity):
  CINIC result is secondary; iteration 03 should study the encode leg (e.g. guided encode, or
  encoding to s* = 1 with fewer/finer steps near s = 1).
- Neither H1 nor H2 gets within ~6 points of the ceiling: report the baseline (102 / 105, 74.8% /
  74.6%, eye ~58% convincing) as the final honest result with its impurity caveat; if time remains,
  iteration 03 = CINIC fine-tune of the cat model for dog->cat symmetry, or combine CINIC-dog + small w.
