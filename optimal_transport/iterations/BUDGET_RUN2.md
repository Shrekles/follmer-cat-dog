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
