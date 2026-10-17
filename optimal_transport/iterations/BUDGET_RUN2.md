iter | job | GPU-min | running total (cap 600)
---|---|---|---
04 | smoke_train (30 steps + tiny snapshot eval, cat; container 0.84 min + ~0.5 startup) | 1.4 | 1.4
04 | smoke_eval (8-image composed ODE + figures; container 0.36 min + ~0.5 startup) | 0.9 | 2.3
04 | train_cat (cancelled by a client-side cancellation at 66.9 train-min, step ~17,000; snapshots s3071/s6140/s12129 kept; container ~68 min + startup) | 68.5 | 70.8
04 | train_dog (cancelled at 65.2 train-min, step ~17,000; snapshots s3108/s6195/s12386 kept) | 68.5 | 139.3
04 | snapeval (per-snapshot stats on the 6 surviving snapshots; container 2.42 min + ~0.5 startup) | 2.9 | 142.2
04 | resume_cat (40 min from cat_s12129 -> s22457 + stats on 4 new snapshots; container 41.70 min + ~0.5 startup) | 42.2 | 184.4
04 | resume_dog (40 min from dog_s12386 -> s22697 + stats; container 41.73 min + ~0.5 startup) | 42.2 | 226.6
04 | eval (composed ODE 512+512, 200 Heun; standard eval; flow/steering/best-pairs/same-noise figures; container 7.03 min + ~0.5 startup) | 7.5 | 234.1
05 | smoke (20 resume steps cat + snap eval n16/5 steps + eval05 n8/10 steps, all figures; container 1.69 min + ~0.5 startup) | 2.2 | 236.3
05 | train_cat (container A: 45-min cosine resume 2e-4->2e-6 from iter04/cat_last.pth, 11,666 steps, + snapshot eval of 6 snapshots at n512/200 steps; container 53.69 min + ~0.5 startup) | 54.2 | 290.5
05 | train_dog (container B: same for dog, 11,723 steps; container 53.48 min + ~0.5 startup) | 54.0 | 344.5
05 | eval (composed ODE 512+512 with iter05 picks, +/- Tweedie; mixed pairs M1-M4; FT/iter04 re-scored +/- Tweedie; latent Gaussianity; all figures; container 19.97 min + ~0.5 startup) | 20.5 | 365.0
06 | smoke 1 (failed at the first figure, empty grid at n=8; container 0.34 min + ~0.5 startup) | 0.8 | 365.8
06 | smoke 2 (tiny ft_full + tiny repick, all code paths; container 0.91 min + ~0.5 startup) | 1.4 | 367.2
06 | A: ft (FT composed ODE on test 512-999 both directions + 1,000-image eval; container 19.82 min + ~0.5 startup) | 20.3 | 387.5
06 | B: repick (stages 1-2 on 64 probe images, 11 encoders/11 decoders per class; stage 3 n=512 + eval + figures; container 24.11 min + ~0.5 startup) | 24.6 | 412.1
