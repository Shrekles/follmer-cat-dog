# Run 2 brief (read this first, then the two files it names)

**Working copy:** `/Users/rexouyang/follmer-runs/follmer-cat-dog`, a git clone on branch
`ot-from-scratch`, outside iCloud. Run every command from there. **Do not touch the old copy at
`~/Desktop/courtade_research_proj`.**

Then read:
1. `optimal_transport/AGENT_CONTEXT.md`: the hard rules, the math and the code layout. Its paths
   point at the old Desktop copy; use this clone instead. There is no iCloud here, so ignore the
   `brctl` notes.
2. `optimal_transport/CLOUD_HANDOFF.md`, sections "Goal", "Where things stand", "Assets",
   "Keep iterating" and "Required figures for every result". Its "Overrides … for the cloud"
   section does not apply: we are on the Mac again, so AGENT_CONTEXT's no-local-GPU-work rules
   apply. The git rule below does apply.

## Run-2 rules (these override anything older)

- **Fresh budget: 600 A10G GPU-minutes** for run 2 (iterations 04 and up). A100 minutes count
  double, and at most 2 Modal containers run at once. Record every job's GPU minutes and keep a
  running total in `optimal_transport/iterations/BUDGET_RUN2.md` as `iter | job | GPU-min |
  running total`. Do not start a job that would push the total past 600.
- **The Mac disk has only about 600 MB free.** Never download `.pth` files, keep downloads under
  about 10 MB per iteration, and never write large files into the clone.
- **Git:** after each iteration, commit code, reports, figures and small JSON on branch
  `ot-from-scratch` and `git push origin ot-from-scratch`. Never push to `main`, never
  force-push, and never commit checkpoints, data, or tensors over 10 MB.
- **Volume paths:** write new outputs under `/vol/iter04/`, `/vol/iter05/` and so on. Never write
  to `/vol/out/`, `/vol/iter0{1,2,3}/` or `/vol/judge/`.
- **Required figures, for every model result:** the flow grid in both directions, the
  steering-strength grid, and the best pairs (see CLOUD_HANDOFF). Also keep the standard eval:
  cost both clamped and unclamped, judge target rate against its ceiling, KID against its floor,
  a figure of the lowest target-probability outputs, and the memorization check.

## Why from scratch, and what to watch

The fine-tuned models carry other CIFAR classes over from the all-class pretrained model.
Models trained from random initialization on one class never learn those classes.

The hope is that the composed ODE still yields same-scene pairs between two **independently**
trained models. Published work reports that diffusion models trained separately, even on
disjoint data, map the same noise to nearly the same image when they generalize: Kadkhodaie et
al., ICLR 2024, on geometry-adaptive harmonic representations; Zhang et al., 2023, on
reproducibility in diffusion models. That consistency breaks down when a model **memorizes** its
training set, and 5,000 images per class is small. So:
- favour smaller networks, augmentation and early-stopping checkpoints;
- always run the memorization check (nearest-training-image distance, computed in
  `tasks_iter03.py`);
- report whether same noise leads to the same scene across the two new models.
