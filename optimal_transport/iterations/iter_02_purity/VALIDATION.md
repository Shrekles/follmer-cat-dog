# Validation of iteration 02 (purity), and the spec for iteration 03

Validator agent, 14:45 PDT 2026-10-15. I read REPORT.md, metrics.json, the GPU logs, `tasks_iter02.py` /
`modal_iter02.py`, and `separate_models/finetune.py`, and looked at `cat_to_dog_compare_big.png`,
`dog_to_cat_compare_big.png`, `prior_same_z_big.png`, `prior_dog_lowest_ptarget_big.png`,
`cat_to_dog_w0.5_lowest_ptarget_big.png` and `cat_to_dog_cinic_w0_lowest_ptarget_big.png`. I ran nothing on the GPU.

## 1. What the figures show

**cat->dog compare (rows: input / baseline / CINIC w=0 / guided w=0.5 / guided w=1.5; columns 0-15).**
- *Baseline:* about 11 of 16 are clear same-scene dogs (1, 2, 4, 6, 7, 8, 10, 11, 12, 14, and 9 if you accept a
  fox-like dog). Column 3 (the yawning cat) is unchanged, column 0 is a messy cat, 13 hardly changes (a dark
  frame), and 5 and 15 are white blobs.
- *CINIC w=0:* almost the same picture as the baseline in most columns, because it decodes the same latents.
  It is worse where it differs. Column 0 is more cat, 3 is still a cat, 5 becomes a garbled table and window
  scene, and 14 is messier. The degradation is real and fits the numbers (judge 47%, KID 14.3).
  **The lowest-p(dog) CINIC outputs show how it fails:** about 9 of 16 are cats left unchanged, and several are
  turned into airplanes, ships or birds (columns 0, 1, 8 and 12 of that figure). That is off-class,
  ImageNet-style content, not dogs.
- *Guided w=0.5:* it fixes several columns that stayed cat (0 gets a dog face, 13 a small dog in the field, 15
  a white dog), and it keeps the layout and colours in most columns (1, 2, 4, 6, 7, 8, 10, 11, 12, 14). But in
  some it replaces the scene with a centred close-up dog: column 5 is a frontal white dog face with the room
  gone, and column 9 has a new pose. The cost doubling (101.6 to 187) matches this: the scene survives in most
  columns but the animal is redrawn larger. Its lowest-p figure shows that **guidance away from "cat" does
  not remove the other non-dog modes**: frog, horse, horse, red car and a green blob remain.
- *Guided w=1.5:* almost every column is a new portrait-style dog, and the scene is lost. This is
  mode-seeking (judge 96%, above the ceiling) and KID 12.7. It is useless.

**dog->cat compare.** In the baseline about 13 of 16 are cats. Column 2 (a dog in a yard) stays a dog, and
11 is ambiguous. w=1.0 fixes column 2 but changes the framing in places (column 14 becomes a cat inside a
circle). The pattern matches cat->dog.

**Prior samples from the same z (cat / dog / CINIC-dog models).** The two CIFAR models put the same layout,
colours and background on the same z, which is what makes the composed ODE cheap. Some of the cat model's
samples look like dogs (0, 14, 15), and some of the dog model's look like cats (8, 11).
**`prior_dog_lowest_ptarget_big.png` is the key image.** The dog model's 16 least dog-like samples are a
boat, a frog, a truck or house, a horse, deer (×3), an aircraft or ship, a cat, and so on. **These are other
CIFAR-10 classes left over from the all-class pretrained initialisation.** The dog model has not fully
forgotten the other classes, and ~11% of its mass is judged cat. The transport outputs fail the same way
(horse, frog and car in the w=0.5 lowest-p figure).

## 2. Verdict on iteration 02

- **The measurements are correct and correctly interpreted.** Costs, judge rates and KID agree with the
  figures, and the guided rows are correctly labelled as not couplings. The tilted law p_tgt^(1+w)/p_src^w is
  not ν, so these rows bound nothing. This includes w=0.5, despite its good judge rate and KID. Note that its
  KID 1.89 is within one se of the baseline's 2.92.
- **CINIC fine-tuning failed as run.** It made both the judge rate and KID much worse, with outputs moving
  toward ImageNet style and other classes. Drop it for the rest of the session: no time remains to try a
  smaller share.
- **The useful finding is diagnostic.** For each model, the target rate of its own prior samples (79.5% dog,
  80.1% cat) is only ~5 points above its transport outputs (74.8% / 74.6%), and ~8 points below the ceilings
  (87.7% / 85.1%). So **most of the purity gap is in the target model's own sampling law, which is the
  target marginal of our coupling.** By eye that gap is mostly other CIFAR-10 classes the model still keeps.
  No decoding trick that keeps T a coupling can fix this. Only a better model of ν can.
- **The best valid coupling remains the iteration-01 composed ODE:** cost 101.6 ± 3.4 (cat->dog) and 104.7 ± 3.4
  (dog->cat), judge 74.8% / 74.6% (ceilings 87.7 / 85.1), KID×1e3 2.9 / 2.2 (floors 0.36 / 0.09). It is
  still not fully valid: its second marginal is measurably off ν, by about 13 judge points and KID ~2.5
  above the floor.

## 3. Choice of the final iteration

| option | improves a *valid* coupling? | fits ~70 min wall? | verdict |
|---|---|---|---|
| (a) continue CIFAR-only fine-tuning of both models, big batch, then decode the saved latents | **yes, directly**: it sharpens ν̂ (removes leftover other-class modes); the decoder fixes the target marginal | yes (two parallel containers ~55 min) | **CHOSEN** |
| (b) small-w guidance sweep 0.1-0.3 | no: tilted law, so it is diagnostic only, and w=0.5 already maps out this trade-off | yes, cheap | skip; it adds nothing toward a valid coupling |
| CINIC with a smaller share / CIFAR annealing | maybe, but two training stages do not fit | no | skip |
| guidance against the all-class model (s_dog + w(s_dog − s_all)) | no: also a tilted law | yes | skip |

Option (a) is the only one that improves the second marginal while keeping T a map whose target is the
model's ν̂. The decode leg is the same ODE, so the cost should stay close to ~100. The prior samples from the
same z show that layouts depend mostly on z, not on the fine details of the model.

**One design point (it avoids any cross-container dependency):** each container fine-tunes only the
*target* model for one direction. It then decodes the latents **already saved in iteration 02**, which the
unchanged source models encoded:
- container **D**: new dog model, decoding `/vol/iter02/latents_cat_to_dog.pt` (from `out/cat.pth`): cat->dog.
- container **C**: new cat model, decoding `/vol/iter02/latents_dog_to_cat.pt` (from `out/dog.pth`): dog->cat.

T = decode_new ∘ encode_old is a valid composed map. Its second marginal is the new model's law, as long as
encode_old(source) ≈ N(0, I), which is the same assumption the baseline makes. This also skips the
~5-minute re-encode.

## 4. SPEC: iteration 03 (`iter_03_cifar_refine`)

**Code.** New files `optimal_transport/tasks_iter03.py` and `optimal_transport/modal_iter03.py`, copied from
the iteration-02 pattern (`modal_iter02.py` ships `recovered_src/` when the originals are evicted). Keep that
mechanism, because iCloud eviction is still a risk. `tasks_iter03.py` can `from tasks_iter02 import *` and add
one function:

`finetune_cifar(cls, init, out, mid, bs=256, lr=1e-4, warmup=200, ema_rate=0.999, cap_min=40.0, deadline=None, reserve_min=17.0, seed=3)`
- Copy `finetune_dog_cinic`, with **CIFAR-10 train images of class `cls` only** (`cifar("train", cls)`; never
  test), random horizontal flip, bf16 autocast, grad-clip 1.0, OOM fallback to accumulation, and a fresh Adam.
- `init = "out/{cls}.pth"` (the step-16000 EMA weights; load-only). **Do not use `--resume` or
  `separate_models/finetune.py` with `--out /vol/out`, because that writes `{cls}_last.pth` and `{cls}.pth`
  in `out/` and would overwrite them.** Starting from the EMA weights with a fresh Adam and warmup is what the
  smoke-tested CINIC code already does. `out/*_last.pth` is not needed.
- LR schedule (time-based, since the cap is time-based): warmup 200 steps to 1e-4, constant until 50% of cap,
  then linear decay to 1e-5 at the cap. EMA 0.999 (time constant ~1k steps ≈ 256k images).
- bs 256. The bs-128 peak was 10.4 GB, so 256 is ≈ 20-21 GB and fits an A10G, with the accumulation fallback
  if not. Expect ~1.2 it/s (~315 img/s, as in iteration 02), so a 40-min cap gives ≈ 2,900 steps ≈ 0.75M
  images. That is ~150 more epochs of 5k images, on top of the 1.02M images seen so far.
- Save `iter03/{cls}_c3_mid.pth` at half the cap and `iter03/{cls}_c3.pth` at the end (slim EMA weights, using
  `_fresh` names). Log the loss every 100 steps.

**Chains (two A10G containers, in parallel):**
- **D (dog):**
  1. `finetune_cifar("dog", init="out/dog.pth", out="iter03/dog_c3.pth", mid="iter03/dog_c3_mid.pth", deadline=D)`, then `vol.commit()`.
  2. `prior_samples({"dog_c3": "iter03/dog_c3.pth"}, n=512, folder="iter03", deadline=D)`. It uses the same
     z (seed 0) as iteration 02, so it can be compared row-for-row with `prior_dog`.
  3. `guided_decode("cat_to_dog", 0.0, tgt_ckpt="iter03/dog_c3.pth", name="iter03_c3", folder="iter02", deadline=D)`.
     It reads the saved latents and writes the new, non-overwriting file
     `/vol/iter02/transport_cat_to_dog_iter03_c3.pt`. If you prefer outputs under `/vol/iter03/`, add a
     `lat_folder` parameter in a copied function. Either is fine; just do not overwrite anything.
  4. Optional, only if `_left(D) > 20` min: `prior_samples({"dog_c3_mid": "iter03/dog_c3_mid.pth"}, ...)`.
     This gives a dose-response point.
  5. `eval_runs([...prior_dog_c3, cat_to_dog_iter03_c3 (+ mid prior)], "evalD", folder="iter03", kid_refs=False)`.
- **C (cat):** the mirror image, with `finetune_cifar("cat", init="out/cat.pth", ...)`, `prior_cat_c3`,
  `guided_decode("dog_to_cat", 0.0, tgt_ckpt="iter03/cat_c3.pth", name="iter03_c3", folder="iter02")`, and `evalC`.
- **Smoke first** (rule 4): `finetune_cifar` with `max_steps=20, bs=256` (this checks memory at the full batch
  size), a 10-step prior with n=4, a 10-step decode of 4 latents, and eval. Use one container, ~2 GPU-min.
- **Memorisation guard (cheap, recommended inside eval or as a tiny extra task):** for the 512 new prior
  samples, compute the median over samples of the min-over-train-images squared L2 distance to the 5,000
  CIFAR-train images of that class. Compute the same statistic for the first 512 real *test* images of that
  class and for the old model's `prior_{cls}.pt`. Flag memorisation if the new median is < 0.5× the
  test-image median. KID against the held-out test set will not catch copying.

**GPU and wall-time plan** (A10G, each container: ~2 min startup, ≤ 40 min training, ~5.5 min prior, ~5 min
decode, ~1 min eval, plus the optional mid prior of ~5.5 min):

| time PDT | event |
|---|---|
| ≤ 14:58 | code written, smoke launched (~2 GPU-min) |
| ≤ 15:03 | C and D launched with `GPU_DEADLINE_EPOCH` = 16:05 PDT; `reserve_min=17` guarantees training stops by 15:48 even if the launch is late |
| ~15:45 | training ends (cap 40 min, or earlier because of the deadline) |
| ~15:57 | prior + decode + eval done (optional mid prior: ~16:03) |
| ≤ 16:20 | download the small results (<10 MB), write figures (`compare` rows: input / baseline / iter03 for both directions; prior_same_z with the old and new models; lowest-p), and write REPORT.md and LOG.md |

GPU: 2 × ~55-60 min plus 2 min smoke ≈ **115-120 GPU-min**, so the session total is ≈ 270 of 480. That is
within budget. Launch nothing after 16:05.

**Success criterion (for a VALID coupling; both directions are judged separately):**
1. Target-rate on T(source) **≥ 79%** (≥ +4 points, ≈ 2 se over 74.8% / 74.6%) and ≤ the ceiling + 2 points
   (no mode-seeking), **and**
2. KID×1e3 on T(source) **≤ baseline** (2.92 c->d, 2.16 d->c), taking the bootstrap se into account, and not
   worse by more than 1 se, **and**
3. clamped cost ≤ 110, **and**
4. the prior samples improve the same way (judge rate up vs 79.5% / 80.1%, KID ≤ 3.28 / 4.42), and the
   memorisation guard is not triggered, **and**
5. by eye, at least as many same-scene dogs (cats) as the baseline in the 16-column compare figure, with
   fewer non-dog-class outputs in the lowest-p figure.

If 1-5 hold in a direction, that direction's iteration-03 map becomes the **new best valid coupling**. If the
judge rate improves but KID gets worse by more than 1 se, report it as a trade-off, not a win.

**Fallback.**
- If bs 256 OOMs even with accumulation, or the smoke fails at bs 256, use bs 128 (known to work).
- If only one container can run (a launch problem or cancellation), run **D (cat->dog)**, which is the
  headline direction.
- If a container is cancelled during training, the `mid` snapshot still exists: run prior + decode + eval on
  it in a short job, but only if it can start before 16:05.
- If nothing improves, the session's final answer stays **the iteration-01 composed ODE: 101.6 / 104.7,
  judge 74.8% / 74.6% vs ceilings 87.7 / 85.1, KID 2.9 / 2.2**. State honestly that its second marginal is
  ~13 judge points and ~2.5 KID units away from ν, so it is an approximately valid coupling: its cost ~100
  is a near-upper bound on W₂², not a proven one.
