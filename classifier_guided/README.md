# Experiment 1: one model + classifier guidance

A pretrained diffusion model, run as a map from a **Brownian path** to an **image**. This is
used to try the time-reversal coupling from Courtade–Fathi–Mikulincer,
[*Stochastic proof of the sharp symmetrized Talagrand inequality*](https://arxiv.org/abs/2407.09465),
on real image distributions (CIFAR-10 cats and dogs).

![cat to dog blends](results/blend_w6/blend_w6_big.png)

*Each row is one Brownian path. Left column is a cat, right column a dog, with 5 blends in
between. Every image in a row is driven by the same path, so pose, layout and colors stay
fixed while the animal changes.*

## How it works

1. **Input: a Brownian path B** on [0, 1]. This is the only randomness.
2. **Run the SDE** (the Föllmer process). Start at X₀ = 0 and take a few hundred small steps:

       dX_t = v(X_t, t) dt + dB_t,     v(x, t) = (E[X₁ | X_t = x] − x) / (1 − t)

   X₁ is the image. The same path always gives the same image.
3. **The drift comes from a pretrained network.** The network predicts a clean image from a
   noisy one. Its noise schedule is converted to Föllmer time by matching signal-to-noise
   ratios, t/(1 − t), and Tweedie's formula gives E[X₁ | X_t].
4. **Cat or dog** comes from a small classifier trained on noisy images. At every step it adds
   a push `w · ∇log p(class | image)`.
5. **Couplings:** run the cat and dog SDEs on related paths: the same B, the time reversal
   B̂_t = B₁ − B₁₋ₜ (the paper's coupling), −B, or an independent path. Then compare
   E|X − Y|². The time grid is symmetric, so B̂ is exactly B's increments in reverse order.

## What came from where

| Source | Used for |
|---|---|
| **The paper** | The framing (sample = function of a Brownian path, via the Föllmer process); the B vs B̂ coupling and the comparison couplings; the quantities to compare, E\|X−Y\|² against W₂² and 2D(μ‖γ)+2D(ν‖γ); the Gaussian extremal case used as a test |
| **[score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)** (git submodule) | The pretrained CIFAR-10 VP model: weights, architecture and noise schedule. Its sampler is not used |
| **This repo** | Conversion of the network's output into the Föllmer drift; the path-driven sampler with exact time reversal; the noisy cat/dog classifier; a Mac/CPU stub for the repo's CUDA ops |

## Results

**Gaussian check** (`test_gaussian.py`): μ = N(0, 4I), ν = N(0, I/4), per dimension.
The reversed coupling attains the optimum, as the proof predicts.

| Dog driven by | E\|X−Y\|²/dim | Theory |
|---|---|---|
| B̂ (reversed) | 2.234 | 2.250 = W₂² = 2D(μ)+2D(ν) |
| B (same) | 2.754 | 2.771 |
| −B | 5.701 | 5.729 |
| independent | 4.228 | 4.250 |

**CIFAR cats vs dogs** (`run_cat_dog.py`, 16 paths, guidance w = 1, dim 3072):

| Dog driven by | E\|X−Y\|² |
|---|---|
| B (same) | 52 ± 9 |
| B̂ (reversed) | 1238 ± 164 |
| independent | 1641 ± 138 |
| −B | 3035 ± 251 |

Finite-sample OT between the two sets of 16 samples: 1003. The entropy estimates give 2D(μ)+2D(ν) ≈ 44,000.
Images: [`results/cat_dog_w1/`](results/cat_dog_w1). The prediction at t = 0.25 already shows the
final layout (`trajectory_reversed_big.png`), so the early part of the path decides the picture.
That explains why the reversed coupling is close to independent here.

**Guidance scan** (`results/blend_scan/scan_big.png`): w = 3, 6, 10, with cat on the left and
dog on the right on the same path. w = 6 was the lowest value that gave clear cats and dogs.

## Limitations (it's a proof of concept)

- **32×32 CIFAR images**, so they're small and blurry.
- **The classifier is the weak link** (74% accuracy on lightly noised images). Recognizable
  animals need w ≈ 6, which gives *sharpened* cat and dog distributions, not the exact ones
  (only w = 1 is exact). Some "cats" look like frogs.
- **The entropy is effectively infinite for images**, which lie close to a low-dimensional set.
  2D(μ)+2D(ν) is huge and the inequality holds trivially; only comparisons between couplings
  are informative.
- **Small samples** (16 paths), so the numbers are rough.
- **Upgrade:** a class-conditional ImageNet-64 model (openai/guided-diffusion) gives exact cat
  and dog distributions with no classifier. Only a new denoiser wrapper would be needed.

## Files

| File | What |
|---|---|
| `../common/follmer.py` | Brownian paths (reverse, negate), Euler–Maruyama Föllmer sampler, exact Gaussian denoiser |
| `../common/test_gaussian.py` | Gaussian extremal-case check (no downloads, ~10 s) |
| `../common/score_sde_backend.py` | Loads a score_sde checkpoint as a Föllmer denoiser, with classifier guidance |
| `noisy_classifier.py` | Trains the noise-conditional CIFAR-10 classifier |
| `run_cat_dog.py` | Coupling experiment (4 couplings, E\|X−Y\|², energies, OT) |
| `blend_cat_dog.py` | Cat→dog blend grids along shared paths |
| `single_path.py` | One path watched over time: cat on B, dog on B, dog on B̂ |
| `make_viewer.py` | Builds `results/viewer/viewer.html`, the time-slider viewer |
| `../common/prepare_cifar.py` | Downloads CIFAR-10 (Hugging Face mirror) as tensors |
| `checkpoints/cifar10_noisy_classifier.pt` | The trained classifier (15 epochs) |

## Reproduce

Run everything from the repository root.

```bash
git clone --recurse-submodules https://github.com/Shrekles/follmer-cat-dog && cd follmer-cat-dog
pip install torch torchvision ml_collections gdown scipy pandas pyarrow
python common/test_gaussian.py

# pretrained score model (990 MB), from the score_sde_pytorch Google Drive folder
mkdir -p checkpoints/vp/cifar10_ddpmpp_continuous
gdown 1le6lIosRGjnraM-HBOsuA1zcqDV8DB76 -O checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth

# optional: retrain the classifier (a trained one is included)
python common/prepare_cifar.py && python classifier_guided/noisy_classifier.py --epochs 15      # ~70 min on an M2

CKPT=checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth
CLF=classifier_guided/checkpoints/cifar10_noisy_classifier.pt
python classifier_guided/run_cat_dog.py   --ckpt $CKPT --classifier $CLF --n 16 --steps 500 --out classifier_guided/results/cat_dog_w1   # ~18 min
python classifier_guided/blend_cat_dog.py --ckpt $CKPT --classifier $CLF --w 6 --k 7 --n 10 --steps 400 --seed 1 --out classifier_guided/results/blend_w6  # ~11 min
```

Runs on CUDA, Apple MPS or CPU.
