"""Fine-tune the pretrained CIFAR-10 score model on ONE class only (e.g. cats, or dogs).

The result is an unconditional model of that single class: no labels, no classifier.
Training uses the same objective as score_sde_pytorch for VP models: noise an image to a
random time s and regress the network output onto the noise that was added.

    python separate_models/finetune.py --cls cat --steps 3200
    python separate_models/finetune.py --cls dog --steps 3200
    python separate_models/finetune.py --cls cat --steps 3200 --resume     # continue after an interruption

Outputs (in separate_models/checkpoints/):
    {cls}_last.pth   full training state (for --resume)
    {cls}.pth        final EMA weights, loadable with common/score_sde_backend.load_score_model
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import argparse
import time

import torch

from follmer import BrownianPath, follmer_sample, symmetric_grid
from score_sde_backend import ScoreSDEDenoiser, get_device, import_score_sde, load_model

CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls", required=True, choices=CIFAR)
    p.add_argument("--repo", default="score_sde_pytorch")
    p.add_argument("--config", default="vp/cifar10_ddpmpp_continuous")
    p.add_argument("--init", default="checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth")
    p.add_argument("--data", default="data/cifar10_train.pt")
    p.add_argument("--out", default="separate_models/checkpoints")
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=0, help="also keep EMA weights every N steps")
    p.add_argument("--preview-n", type=int, default=16, help="samples drawn at the end")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)
    last, final = f"{args.out}/{args.cls}_last.pth", f"{args.out}/{args.cls}.pth"

    config, sde, eps, model = load_model(args.repo, args.config, args.init, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = [p.detach().clone() for p in model.parameters()]
    step = 0
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        ema = [s.to(device) for s in ck["ema"]["shadow_params"]]
        step = ck["step"]
        print(f"resumed from step {step}")

    d = torch.load(args.data)
    x_all = d["x"][d["y"] == CIFAR.index(args.cls)].to(device)          # uint8, (N, 3, 32, 32)
    print(f"device={device}  class={args.cls}  images={len(x_all)}  steps={args.steps}  bs={args.bs}  lr={args.lr}", flush=True)

    def save(path, slim=False):
        if slim:  # EMA weights only, in the layout load_score_model expects
            backup = [p.detach().clone() for p in model.parameters()]
            for p_, e in zip(model.parameters(), ema):
                p_.data.copy_(e)
            torch.save({"model": model.state_dict(), "ema": {"shadow_params": []}, "step": step, "cls": args.cls}, path)
            for p_, b in zip(model.parameters(), backup):
                p_.data.copy_(b)
        else:
            torch.save({"model": model.state_dict(), "ema": {"shadow_params": [e.cpu() for e in ema]},
                        "optimizer": opt.state_dict(), "step": step, "cls": args.cls}, path + ".tmp")
            os.replace(path + ".tmp", path)

    model.train()
    t0, running = time.time(), 0.0
    while step < args.steps:
        idx = torch.randint(len(x_all), (args.bs,), device=device)
        x = x_all[idx].float() / 255
        flip = torch.rand(args.bs, device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        x = x * 2 - 1 if config.data.centered else x
        s = torch.rand(args.bs, device=device) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        mean, std = sde.marginal_prob(x, s)
        out = model(mean + std[:, None, None, None] * z, s * 999)         # network predicts the noise
        loss = (out - z).pow(2).mean()
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for e, p_ in zip(ema, model.parameters()):
                e.mul_(args.ema).add_(p_.detach(), alpha=1 - args.ema)
        step += 1
        running += loss.item()
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step}/{args.steps}  loss={running / 50:.4f}  {el / 60:.0f} min elapsed", flush=True)
            running = 0.0
        if step % args.save_every == 0 or step == args.steps:
            save(last)
        if args.snapshot_every and step % args.snapshot_every == 0:
            save(f"{args.out}/{args.cls}_step{step}.pth", slim=True)

    save(final, slim=True)
    print(f"saved {final}", flush=True)

    # preview: samples from the fine-tuned EMA model, via the Föllmer sampler
    import torchvision
    _, mutils = import_score_sde(args.repo)
    for p_, e in zip(model.parameters(), ema):
        p_.data.copy_(e)
    model.eval()
    den = ScoreSDEDenoiser(sde, eps, mutils.get_score_fn(sde, model, train=False, continuous=True))
    shape = (args.preview_n, config.data.num_channels, config.data.image_size, config.data.image_size)
    res = follmer_sample(den, BrownianPath(symmetric_grid(250), shape, seed=123), device)
    os.makedirs("separate_models/results", exist_ok=True)
    torchvision.utils.save_image(((res.x1 + 1) / 2).clamp(0, 1), f"separate_models/results/preview_{args.cls}.png", nrow=8, padding=1)
    torch.save(res.x1, f"separate_models/results/preview_{args.cls}.pt")
    print(f"wrote separate_models/results/preview_{args.cls}.png", flush=True)


if __name__ == "__main__":
    main()
