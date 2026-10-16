# Briefing for the training and validator agents

You are one of two agents iterating on Experiment 3 of a research project. Read this whole file
before doing anything. The repository root is `/Users/rexouyang/Desktop/courtade_research_proj`.
Run every command from there.

## The research question

The project is about the sharp symmetrized Talagrand inequality (Courtade–Fathi–Mikulincer,
arXiv:2407.09465). For μ centered,

    W₂²(μ, ν) ≤ 2 D(μ‖γ) + 2 D(ν‖γ)

where W₂² is the optimal squared transport cost: the minimum of E|X − Y|² over couplings (X, Y)
with X ~ μ and Y ~ ν. Here μ = CIFAR-10 cats and ν = CIFAR-10 dogs (32×32×3, pixels in [−1, 1],
dimension 3,072).

The proof couples μ and ν through Föllmer processes driven by a Brownian path B and its time
reversal B̂. Experiment 2 showed that this coupling gives unrelated-looking pairs (cost ≈ 1107).
**Experiment 3 instead looks for the optimal coupling: a map T with T(X) ~ ν that makes
E|X − T(X)|² as small as possible.** The ideal output is a cat and a dog that look very alike
(same pose, background and colors) while each is clearly its own animal, with small transport
cost.

**Rules that keep it honest:**
- A coupling only bounds W₂² if its second marginal is really ν: the outputs T(X) must look like
  real dogs. A map that barely changes the image has tiny cost and is useless. **Always report
  cost together with how dog-like (or cat-like) the outputs are.**
- Use real CIFAR-10 test images as inputs, and keep test images out of all training.

## What exists

**Models.** Each is the score_sde_pytorch "DDPM++ continuous VP" U-Net (61.8M parameters):
- **pretrained all-class model:** `checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth`
- **cat-only and dog-only models:** fine-tuned from the pretrained one for 16,000 steps (batch 64,
  lr 1e-4 after step 4,000).
  - local copies: `separate_models/checkpoints/{cat,dog}.pth`
  - on the Modal volume: `out/cat.pth`, `out/dog.pth`
  - EMA snapshots on the volume: `out/{cat,dog}_step{8000,12000,16000}.pth`
  - full training state for resuming: `out/{cat,dog}_last.pth`
  - first 4,000-step versions: `out/v1_4k/{cat,dog}.pth`

**Composed ODE (the current best method).** Each model has a probability-flow ODE between images
and noise. `T = decode_dog ∘ encode_cat` maps a cat to a dog (`optimal_transport/pf_ode.py`:
`encode`, `decode`, `transport`; Heun solver, 200 steps, eps = 1e-3).

**Baseline results** (`optimal_transport/results/`, `optimal_transport/README.md`), on the first
512 test cats and the first 512 test dogs:

| Coupling | Cost |
|---|---|
| Composed ODE, cat → dog | 102 |
| Composed ODE, dog → cat | 105 |
| Encode, then decode with the same model | ≈ 0.003 |
| Learned Brenier flow (exact OT on 5,000 train pairs + flow matching) | 245, but outputs blurry, a failure |
| Exact optimal assignment, real train photos (5,000 ↔ 5,000) | 592 |
| Exact optimal assignment, real test photos (1,000 ↔ 1,000) | 660 |
| Föllmer, same path B | 129 |
| Föllmer, reversed path (the proof's coupling) | 1107 |

Visually, most cats become same-scene dogs, but some barely change. The only classifier so far is
`classifier_guided/checkpoints/cifar10_noisy_classifier.pt` (`separate_models/purity.py`). It is
too weak to judge purity: it calls only 56% of real cats "cat". **A stronger judge is needed.**

**Gaussian check** (`optimal_transport/gaussian_check.py`): with exact scores, the composed ODE
equals the Brenier map whenever the covariances commute.

## Code you can reuse

- `common/score_sde_backend.py`: `load_model(repo, config, ckpt, device)` returns
  `(config, sde, eps, network)`; `load_score_model(...)` returns a score function.
- `optimal_transport/tasks.py`: GPU job functions (`ddib`, `ot_flow`).
- `optimal_transport/modal_ot.py`: the Modal app template (image, volume, `run(task, kwargs)`,
  `--smoke`). Copy this pattern.
- `separate_models/finetune.py` and `modal_train.py`: single-class fine-tuning on Modal.

Inside a container, the repo lives at `/root/proj` and the volume at `/vol`.
- Pretrained model: `/vol/checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth`
- Data: `/vol/data/cifar10_{train,test}.pt`, dicts of uint8 tensors `x: (N, 3, 32, 32)` and
  `y: (N,)`. Classes: cat = 3, dog = 5.

## Larger datasets (allowed; the user asked for this)

CIFAR-10 has only 5,000 training images per class. You may train on larger cat and dog datasets
when the validator judges it worthwhile:

- **CINIC-10** is the most direct option: 32×32, the same 10 classes as CIFAR-10, about 27,000
  images per class (CIFAR images plus downsampled ImageNet images).
- **Other options:** downsampled ImageNet at 32×32 (dog breeds and domestic-cat classes), LSUN
  cat/dog, or AFHQ downsampled to 32×32.

Rules for datasets:
- **Download inside a Modal container straight to the volume** (for example `/vol/data/cinic10/`),
  never to this Mac. Build uint8 tensor files there in the same format as `cifar10_train.pt`.
- Use only public research datasets from their official or well-known mirrors, and record the
  source URL in `PLAN.md`.
- Keep the CIFAR-10 test images out of training. They stay the evaluation set, so later
  iterations stay comparable. If a dataset contains CIFAR-10 test images (CINIC-10 includes CIFAR
  images), exclude them.
- Note in `REPORT.md` that ImageNet-derived images differ somewhat in style from CIFAR's.

## Hard rules

1. **No training or heavy inference on this Mac** (Apple M2: about 0.14 s per image per training
   step). All GPU work goes through Modal. Local work is limited to small analysis: figures, JSON,
   and a classifier pass over at most about 1,000 images.
2. **The disk is almost full** (about 600 MB free). The project folder syncs to iCloud, which offloads files when space runs low: they still appear in a listing but reading them stalls. Bring them back with `brctl download <path>`. Keep downloads to the Mac under about 10 MB per iteration. Never download model checkpoints. Leave
   `.pth` files on the Modal volume and download only small results (metrics JSON, figures,
   tensors under about 30 MB per iteration).
3. **Modal budget:** use A10G GPUs (`gpu="A10G"`), at most 2 containers at once, and at most 8
   GPU-hours across the whole session (all iterations together, about $9). Record the GPU minutes
   of every job. If Modal reports a spending or usage limit, stop and report it; do not retry.
   **Use large batches** (the user asked for this) to keep the GPU busy:
   - **Training:** batch ≥ 128 (256 if it fits). Scale the learning rate sensibly; use bf16
     autocast or gradient accumulation if memory is tight.
   - **Inference and evaluation:** batches of 256–512.
   - **If a large batch doesn't fit an A10G (24 GB):** an A100 is allowed. Count its minutes at
     2× toward the 8 GPU-hour cap.
4. **Launching Modal:**
   `PYTHONUNBUFFERED=1 nohup modal run --detach <file>.py ... > logs/<name>.log 2>&1 &`
   Watch the log. Once, a client sat idle before starting any container. If a log shows no
   progress for 5 minutes, kill the client and relaunch. Always run a `--smoke` version (tiny
   sizes) of new GPU code first.
5. **Never delete or overwrite** existing results, checkpoints, or volume files. Write new files
   with iteration-specific names. Do not change existing functions in `common/` in ways that break
   their current callers. Adding new files or functions is fine.
6. **Never** commit, push, or touch git, and never modify files outside this repository.
7. **The deadline is absolute** (given in your task). Do not start GPU jobs that cannot finish
   before it.

## Iteration bookkeeping

Each iteration gets a folder `optimal_transport/iterations/iter_NN_<short-name>/`.
- **Training agent** writes `PLAN.md` (what and why), launches the jobs, and puts the results
  there: `metrics.json`, `*_big.png` figures enlarged 4× with `PIL.Image.NEAREST`, and a short
  `REPORT.md`.
- **Validator** writes `VALIDATION.md`.
- **Both** append one line to `optimal_transport/iterations/LOG.md`:

      iter | what | GPU min | cost c→d / d→c | judge dog-rate on T(cats) | verdict

**Standard evaluation:**
- **Inputs:** the first 512 test cats and the first 512 test dogs.
- **Metrics:** mean E|x − T(x)|² in each direction, with standard error; the strong judge's rate of
  the target class on T(x), compared with its rate on real test images of that class; plus a
  16-column figure with rows input / output.
