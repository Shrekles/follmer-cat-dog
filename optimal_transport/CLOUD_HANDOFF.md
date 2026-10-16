# Cloud handoff: continue the cat/dog optimal-transport experiment

You are continuing a research project in a cloud session. The repository is your working copy
(run everything from its root). Read these first, in order:

1. `optimal_transport/AGENT_CONTEXT.md`: the goal, the math, what exists, and the rules.
   It was written for a Mac; the overrides below take precedence.
2. `optimal_transport/iterations/SUMMARY.md` and `iterations/LOG.md`: what the last 4-hour loop
   tried and found.
3. `optimal_transport/README.md`: Experiment 3 baseline results and the Gaussian check.

## Goal (unchanged)

Build a valid coupling of CIFAR-10 cats μ and dogs ν: the outputs must really be distributed like
real dogs (and cats), with E|X − T(X)|² as small as possible, so cat/dog pairs look nearly
identical apart from the animal. This approaches the Brenier map, the optimal coupling behind
W₂² in the sharp symmetrized Talagrand inequality. Always report cost together with how
convincing the outputs are (judge target rate against the real-image ceiling, KID against the
real-data floor, and a visual check). A map that barely changes images is not progress.

## Where things stand

**Best so far:** the composed probability-flow ODE, T = decode_dog ∘ encode_cat, with the
fine-tuned single-class models `out/cat.pth` and `out/dog.pth` on the Modal volume.

| Direction | Cost | Judge target rate (ceiling) | KID ×1e3 |
|---|---|---|---|
| cat → dog | 101.6 | 74.8% (87.7%) | 2.9 |
| dog → cat | 104.6 | 74.6% (85.1%) | 2.2 |

By eye, 58% are convincing same-scene translations.

**Diagnosis:** both models still emit other CIFAR classes (boat, frog, truck, horse, deer),
inherited from the all-class pretrained model they were fine-tuned from. Further CIFAR
fine-tuning barely moved them, CINIC-10 ImageNet data made them worse, and guidance tricks change
the target law, so they are not couplings.

**Recommended next step:** cat-only and dog-only models trained **from scratch**, so they never
learn the other classes. Weigh a smaller U-Net for speed against the budget, and watch for
memorization (only 5,000 training images per class; a memorization check exists in
`tasks_iter03.py`). Then evaluate the composed ODE with the standard harness, using the
iteration-02/03 code and the saved latents.

## Assets (all on the Modal volume `follmer-cat-dog`; nothing large is in git)

| What | Volume path |
|---|---|
| Pretrained all-class model | `checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth` |
| CIFAR-10 | `data/cifar10_{train,test}.pt` (uint8; cat = 3, dog = 5) |
| CINIC-10 tensors | `data/cinic10*` |
| Fine-tuned single-class models (current best) | `out/{cat,dog}.pth`, plus `out/{cat,dog}_step*.pth` snapshots and `out/v1_4k/` |
| Strong judge (ResNet-18, 93%) | `judge/cifar10_judge.pth` |
| Saved latents of the first 512 test cats and dogs (cat/dog encoders) | `iter02/` |
| Iteration-3 refined models | `iter03/{cat,dog}_c3.pth` |

## Overrides to AGENT_CONTEXT.md for the cloud

- **Ignore the Mac-specific parts:** the iCloud and `brctl` notes, the full disk, the
  `/Users/rexouyang/...` paths. Use repo-relative paths. Disk is not tight here, but still never
  download `.pth` files; keep them on the volume.
- **This machine has no GPU.** All GPU work goes through Modal (`modal run --detach ...`); the
  existing `modal_iter0*.py` files show the pattern. Run the GPU code at tiny sizes first.
- **Modal authentication** comes from the environment variables `MODAL_TOKEN_ID` and
  `MODAL_TOKEN_SECRET`. Check with `modal profile current` or `modal volume ls follmer-cat-dog`.
  If authentication fails, stop and say so. Never print or log the token values.
- **Budget:** at most **BUDGET_GPU_HOURS** A10G-hours in total (A100 counts double), and at most
  2 containers at once. If Modal reports a spending or usage limit, stop and report it.
- **Optional agent loop:** dispatch a training agent per iteration, then a validator agent that
  inspects every output figure and number and specifies the next step. Keep the iteration
  bookkeeping (`iterations/iter_NN_*/` with PLAN, REPORT, VALIDATION, and LOG.md lines).
- **Git:** commit your code, reports and small result files (figures, JSON, tensors under about
  10 MB) to a new branch, push it, and summarize. Never commit model checkpoints or data, and
  never force-push or rewrite `main`.

## Keep iterating

Run iteration after iteration (train or adjust, then evaluate, then validate, then decide the next
step) until the GPU budget is used up or the user stops you. Do not stop after one experiment.
The aim is cat/dog pairs that look striking: the same scene, pose and colors, with a clearly
different animal. Each validator review should name the most visually convincing pairs so far
and steer the next iteration toward more of them, without giving up the honesty rules (valid
couplings, cost reported with judge rate and KID).

## Required figures for every result

Alongside the standard evaluation, every iteration that produces or changes a model must save:

1. **ODE flow grid** (`flow_grid_big.png`). For 8 real test cats, one row each:
   - the cat at several points along the cat model's ODE toward noise (s ≈ 0.2, 0.4, 0.6, 0.8, 1.0);
   - then the dog model's ODE back from noise toward an image (s ≈ 0.8, 0.6, 0.4, 0.2);
   - ending at the final dog.

   This makes the cat → noise → dog path visible. Do the same for 8 dogs → cats
   (`flow_grid_dog_to_cat_big.png`).
2. **Steering-strength grid** (`steering_grid_big.png`). For the same 8 cats, one row each, the
   output at increasing steering strength: partial encoding s* = 0.2, 0.4, 0.6, 0.8, 1.0, then
   guidance w = 0.25, 0.5, 1.0 at s* = 1. Label each column with its setting, and print the mean
   cost and judge dog-rate per column under the grid, or in `steering_grid.json`. This shows how
   harder steering trades scene preservation for "dogness". Mark clearly that only s* = 1 with
   w = 0 is a valid coupling; the other columns are diagnostics.
3. **Best pairs** (`best_pairs_big.png`). The 16 most convincing cat → dog pairs of this
   iteration (high judge dog-probability and low cost), input above output, so the visual
   progress is easy to see.

Make all figures upscaled 4× with nearest-neighbour resampling, and keep the GPU cost of these
figures small: 8 images, with latents reused where possible.

## Deliverable

A short report of the best valid coupling found, measured exactly like the table above, with
figures and an honest comparison to the current best. Include GPU-hours used and the Modal volume
paths of any new models.
