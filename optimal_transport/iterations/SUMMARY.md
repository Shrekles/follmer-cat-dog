# Agent loop summary (2026-10-15, 12:35–16:00 PDT)

The goal was a valid coupling of CIFAR-10 cats (μ) and dogs (ν): the outputs must be distributed
like real dogs, cat/dog pairs should look alike, and E|X − T(X)|² should be small. The loop ran
three training iterations, each followed by a validator review. It used about 255 of the 480
allowed A10G GPU-minutes.

## Best model

The composed probability-flow ODE, T = decode_dog ∘ encode_cat, with the 16,000-step single-class
models `out/cat.pth` and `out/dog.pth` on the Modal volume `follmer-cat-dog`. This is the
iteration-01 measurement of the baseline; no later iteration beat it.

| Direction | Cost (clamped) | Judge: target class (real-image ceiling) | KID ×1e3 (real-data floor) |
|---|---|---|---|
| cat → dog | 101.6 ± 3.4 | 74.8% (87.7%) | 2.9 ± 0.9 (0.36) |
| dog → cat | 104.6 ± 3.4 | 74.6% (85.1%) | 2.2 ± 0.6 (0.09) |

By eye, over 64 full-encode columns: 58% are convincing same-scene translations, 30% are left as
the source animal, and 12% are broken. The coupling is therefore only approximately valid: its
dog marginal sits about 13 judge points short of real dogs.

## Iterations

| Iteration | What | Result |
|---|---|---|
| 01 | Strong judge (ResNet-18, 93% test accuracy), evaluation harness, baseline re-measured, partial-encoding sweep | Baseline as above. The class mostly switches by s* = 0.6. Partial encodings are not couplings. |
| 02 | Source of the impurity; dog model fine-tuned with CINIC-10 ImageNet dogs; difference-of-models guidance | The impurity sits in the models' own samples (79.5% / 80.1%). CINIC made things worse (47%, KID 14). Guidance w = 0.5 passes the judge (89.5%) but doubles cost and is not a coupling. |
| 03 | Both models refined on CIFAR only, batch 256, 40 min each | No meaningful change (76.0% / 76.2%, cost 110 / 114). Fails the success criteria. |

## Diagnosis

The cat and dog models still produce samples of other CIFAR classes (boat, frog, truck, horse,
deer), carried over from the all-class model they were fine-tuned from. No decoding trick that
keeps the coupling valid can remove those. Continued fine-tuning barely moved the models, and
ImageNet-derived data hurt. The most promising next step is a dog model (and a cat model) trained
from scratch on dogs only, so it never learns the other classes. That needs several GPU-hours.
