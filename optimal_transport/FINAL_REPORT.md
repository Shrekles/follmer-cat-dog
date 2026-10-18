# Final report: optimal transport between cat and dog diffusion models

**Question.** The sharp symmetrized Talagrand inequality bounds W₂²(μ, ν) by 2D(μ‖γ) + 2D(ν‖γ).
The proof's time-reversed Föllmer coupling (cat model driven by B, dog model by B̂) gives
unrelated-looking pairs (cost ≈ 1107; Experiment 2). So we looked for the **optimal** coupling of
CIFAR-10 cats μ and dogs ν: a map T with T(cat) distributed like real dogs and E|X − T(X)|² as
small as possible, which in practice means cat/dog pairs that look the same apart from the
animal.

**Method.** The composed probability-flow ODE, T = decode_dog ∘ encode_cat: a cat is carried to
noise by the cat model's ODE, then carried out of the noise by the dog model's ODE (and the
reverse for dog → cat). With exact Gaussian scores, this map equals the Brenier map whenever the
covariances commute, including the equality case C, C⁻¹ (`gaussian_check.py`).

## 1. Best valid coupling: the fine-tuned pair, on all 1,000 test images per direction

The models are the 16,000-step single-class models fine-tuned from the pretrained CIFAR-10 model
(Modal volume `follmer-cat-dog`: `out/cat.pth`, `out/dog.pth`). Settings: 200 Heun steps, no final
denoising step.

| Direction | Cost E\|X−T(X)\|² | Judged as the target animal (real-photo ceiling) | KID ×1e3 (real-data floor ≈ 0) |
|---|---|---|---|
| cat → dog | **99.1 ± 2.2** (median 77) | 73.6% (87.7%) | 2.8 ± 0.7 |
| dog → cat | **108.9 ± 2.7** (median 85) | 76.8% (85.1%) | 2.4 ± 0.5 |

By eye on 32 images per direction: 20 (cat → dog) and 22 (dog → cat) are convincing same-scene
translations. The judge is a ResNet-18 with 93% test accuracy, used only to score outputs. For
comparison, the exact optimal assignment between real photos costs 592 (5,000 ↔ 5,000), and the
proof's reversed coupling 1107.

**Validity is approximate.** About a quarter of outputs are not judged the target animal, mostly
the input left unchanged or a leftover object from the all-class pretrained model (boat, horse,
truck), and real images do not encode to exactly Gaussian noise. So 99 / 109 is a strong but not
rigorous upper bound on W₂² between real cats and real dogs.

## 2. Key finding: same noise, same scene, even for independently trained models

Cat and dog models trained **from scratch**, separately, with no shared weights, decode the same
noise to the same scene. The matched/mismatched distance ratio is 0.07 for the independent
models, against 0.06 for the fine-tuned pair (`iterations/iter_04_from_scratch/same_noise_both_models_big.png`).
So same-scene translation is not an artifact of a shared starting model. This matches published
reports that diffusion models which generalize learn nearly the same noise → image map.

## 3. From-scratch models: purer, but softer

Small U-Nets (9.9M parameters), trained from random weights on 5,000 images per class. The best
pair was re-picked by trying every saved snapshot as encoder and decoder (iteration 06), on the
first 512 test images:

| Direction | Coupling | Cost | Judge | KID ×1e3 | Same-scene by eye (of 32) |
|---|---|---|---|---|---|
| cat → dog | fine-tuned | 101.6 | 74.8% | 2.9 | 20 (8 unchanged) |
| cat → dog | from scratch, re-picked | 123.2 | **81.1%** | 8.3 | **23 (1 unchanged)** |
| dog → cat | fine-tuned | 104.6 | 74.6% | 2.2 | **22** |
| dog → cat | from scratch, re-picked | 136.8 | **78.3%** | 4.7 | 13 |

- **Purity:** no leftover other-class objects, and in cat → dog almost no unchanged inputs.
- **Quality:** blurrier images (higher KID) and higher cost.
- **Training length:** these models peak at 60–80 minutes of training, then begin to overfit 5,000
  images. Earlier snapshots make better encoders.
- **The fine-tuned pair stays the best valid coupling**, but the from-scratch cat → dog outputs
  are among the most striking (see the gallery).

## 4. Gallery: most striking pairs

A curated 24-pair gallery, chosen by judge scores plus inspection at 5× zoom, is in
[`gallery/`](gallery/README.md) (`gallery_cat_to_dog.png`, `gallery_dog_to_cat.png`).
The table below is the validator's earlier selection.

Selected by the validator agent. Figures are in `iterations/`.

| # | Pair | Figure |
|---|---|---|
| 1 | tabby face → brown puppy face (fine-tuned, test #10) | `iter_05_decay/compare_cat_to_dog_big.png` |
| 2 | black-and-white cat → Pekingese (fine-tuned, #8) | same |
| 3 | black cat → Boston terrier (from scratch, #13) | same |
| 4 | orange long-haired cat → white fluffy dog (from scratch, #19) | same |
| 5 | dark cat silhouette → Japanese chin (from scratch, #31) | same |
| 6 | cat on a blanket → golden-retriever pup (from scratch, #6) | same |
| 7 | white cat on black → white bulldog (from scratch, #25) | same |
| 8 | orange cat → boxer face (from scratch, #133) | `iter_05_decay/best_pairs_big.png` |
| 9 | sitting brown cat → sitting brown dog (from scratch, #392) | same |
| 10 | shepherd face → green-eyed cat (fine-tuned, dog #4) | `iter_05_decay/compare_dog_to_cat_big.png` |
| 11 | chihuahua in a red hood → cat face (fine-tuned, dog #14) | same |
| 12 | frontal dog → frontal grey tabby (from scratch, #331) | `iter_05_decay/best_pairs_dog_to_cat_big.png` |

Also see `iter_06_repick_full/ft_best_pairs_{cat_to_dog,dog_to_cat}_1000_big.png` (best fine-tuned
pairs over all 1,000 images), `iter_06_repick_full/compare_*_big.png` (input / fine-tuned /
from scratch / re-pick), the steering grid `iter_04_from_scratch/steering_grid_big.png`, and the
flow grid `iter_05_decay/flow_grid_big.png`.

## 5. What did not work

| Attempt | Result |
|---|---|
| Learned Brenier map (exact OT on 5,000 pairs + flow matching) | Blurry blends; does not reproduce its own pairs |
| Extra ImageNet dog data (CINIC-10) | Worse: judge 47%, KID 14 |
| More CIFAR fine-tuning, batch 256 | No change |
| Learning-rate decay on from-scratch models | Worse: KID rises, early overfitting |
| Mixing encoders and decoders across training runs | Fails: each encoder's latents have their own scale, so outputs come out washed out or posterized |
| Guidance by the difference of the two models | Clearly dogs, but 2–4× the cost; not a valid coupling (tilted target law) |
| Final denoising (Tweedie) step | Slightly worse KID in all 10 output sets |

## 6. Compute

- **Run 1** (4-hour agent loop, iterations 01–03): about 255 A10G GPU-minutes.
- **Run 2** (iterations 04–06): 412 of 600 GPU-minutes.
- **Earlier model training** (Experiment 2 fine-tuning and Experiment 3): about 3 GPU-hours.

## 7. Where things are

- **Per-iteration write-ups:** `iterations/iter_NN_*/`, each with PLAN, REPORT and VALIDATION.
- **Logs:** `iterations/LOG.md` and `iterations/BUDGET_RUN2.md`.
- **All models, latents and outputs:** on the Modal volume `follmer-cat-dog`.
  - fine-tuned pair: `out/`
  - from-scratch snapshots: `iter04/`, `iter05/`
  - judge: `judge/`
  - full-test outputs: `iter06/`
- **Re-picked from-scratch pairs:**
  - cat → dog: encoder `iter04/cat_s12129.pth`, decoder `iter04/dog_s20119.pth`
  - dog → cat: encoder `iter04/dog_s17541.pth`, decoder `iter04/cat_s22457.pth`

## 8. Possible next steps

- **Larger single-class datasets at CIFAR's style and resolution.** These would let from-scratch
  models reach the fine-tuned pair's sharpness without overfitting. ImageNet-derived images did
  not work at 32×32.
- **A higher-resolution, class-conditional or per-class pretrained model** (for example
  ImageNet-64 or AFHQ), where both purity and sharpness come built in.
- **Measuring each step of the proof's inequality chain on these models.** The martingale
  integrands F_t = Cov(X₁ | X_t)/(1 − t) are computable with automatic differentiation.
