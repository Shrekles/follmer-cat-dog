"""Run separate_models/finetune.py on Modal GPUs, one container per class, in parallel.

    modal run separate_models/modal_train.py                              # cat and dog, defaults
    modal run separate_models/modal_train.py --steps 4000 --bs 64 --classes cat,dog

What it does:
  1. uploads the pretrained checkpoint and the CIFAR-10 training tensor to a Modal Volume (once)
  2. fine-tunes one model per class on a GPU (same script as the local run)
  3. downloads {cls}.pth into separate_models/checkpoints/ and the sample previews into
     separate_models/results/
Run from the repository root.
"""
import os
import subprocess

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
DATA = "data/cifar10_train.pt"
SCORE_SDE_COMMIT = "cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44"

app = modal.App("follmer-cat-dog")
vol = modal.Volume.from_name("follmer-cat-dog", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "ml_collections", "numpy<2", "scipy", "pillow")
    .run_commands(
        "git clone https://github.com/yang-song/score_sde_pytorch /root/proj/score_sde_pytorch",
        f"cd /root/proj/score_sde_pytorch && git checkout {SCORE_SDE_COMMIT}",
    )
    .add_local_dir(os.path.join(ROOT, "common"), "/root/proj/common", ignore=["__pycache__"])
    .add_local_file(os.path.join(ROOT, "separate_models", "finetune.py"), "/root/proj/separate_models/finetune.py")
)


GPU = os.environ.get("MODAL_GPU", "A10G")  # e.g. MODAL_GPU=T4 modal run ...


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=6 * 3600)
def train(cls: str, steps: int, bs: int, lr: float):
    os.chdir("/root/proj")
    cmd = ["python", "-u", "separate_models/finetune.py", "--cls", cls, "--steps", str(steps), "--bs", str(bs),
           "--lr", str(lr), "--init", f"/vol/{CKPT}", "--data", f"/vol/{DATA}", "--out", "/vol/out",
           "--save-every", "1000", "--snapshot-every", "4000", "--preview-n", "64", "--resume"]
    subprocess.run(cmd, check=True)
    subprocess.run(["cp", f"separate_models/results/preview_{cls}.png", f"separate_models/results/preview_{cls}.pt", "/vol/out/"], check=True)
    vol.commit()
    return cls


@app.local_entrypoint()
def main(classes: str = "cat,dog", steps: int = 4000, bs: int = 64, lr: float = 5e-5):
    have = set()
    for d in ("checkpoints/vp/cifar10_ddpmpp_continuous", "data"):
        try:
            have |= {e.path for e in vol.listdir(d)}
        except Exception:  # noqa: BLE001  (folder not there yet)
            pass
    todo = [p for p in (CKPT, DATA) if p not in have]
    print("on volume:", sorted(have), flush=True)
    if todo:
        print("uploading", todo, flush=True)
        with vol.batch_upload() as up:
            for p in todo:
                up.put_file(os.path.join(ROOT, p), p)

    names = classes.split(",")
    print("starting", names, "on", GPU, flush=True)
    for done in train.starmap([(c, steps, bs, lr) for c in names], order_outputs=False):
        print("finished:", done, flush=True)

    os.makedirs(os.path.join(ROOT, "separate_models/checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "separate_models/results"), exist_ok=True)
    for c in names:
        for remote, local in ((f"out/{c}.pth", f"separate_models/checkpoints/{c}.pth"),
                              (f"out/preview_{c}.png", f"separate_models/results/preview_{c}.png"),
                              (f"out/preview_{c}.pt", f"separate_models/results/preview_{c}.pt")):
            with open(os.path.join(ROOT, local + ".part"), "wb") as f:
                for chunk in vol.read_file(remote):
                    f.write(chunk)
            os.replace(os.path.join(ROOT, local + ".part"), os.path.join(ROOT, local))
            print("downloaded", local, flush=True)
