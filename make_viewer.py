"""Build an interactive viewer (one self-contained HTML file) with a time slider.

For each Brownian path B it shows three processes side by side as t goes from 0 to 1:
    cat driven by B   |   dog driven by the same B   |   dog driven by B̂_t = B_1 − B_{1−t}
in two views: the model's current guess x̂_1(X_t, t), or the raw state X_t.

    python make_viewer.py --ckpt checkpoints/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth \
        --classifier checkpoints/cifar10_noisy_classifier.pt          # samples, then writes the HTML
    python make_viewer.py --html-only                                  # rebuild the HTML from saved frames
"""
import argparse
import base64
import io
import json
import os
import time

import torch
import torch.nn.functional as F

RUNS = ["cat · driven by B", "dog · driven by the same B", "dog · driven by reversed B̂"]


def sample(args):
    from follmer import BrownianPath, StackedPath, follmer_sample, symmetric_grid
    from noisy_classifier import load_classifier
    from run_cat_dog import CIFAR
    from score_sde_backend import ScoreSDEDenoiser, get_device, load_score_model

    device = get_device()
    config, sde, eps, score_fn = load_score_model(args.repo, args.config, args.ckpt, device)
    clf = load_classifier(args.classifier, device)
    den = ScoreSDEDenoiser(sde, eps, score_fn, classifier=clf, guidance=args.w)
    shape = (config.data.num_channels, config.data.image_size, config.data.image_size)
    cat, dog = CIFAR.index("cat"), CIFAR.index("dog")
    n = args.n

    B = BrownianPath(symmetric_grid(args.steps), (n,) + shape, seed=args.seed)
    labels = torch.cat([torch.full((n,), c) for c in (cat, dog, dog)])
    ts = [j / args.frames for j in range(1, args.frames)]
    t0 = time.time()
    res = follmer_sample(den, StackedPath([B, B, B.reversed()]), device, labels=labels, snapshot_ts=ts, verbose=True)
    print(f"sampled in {time.time() - t0:.0f}s")

    # (frames, runs, paths, C, H, W); the last frame is the sample X_1 in both views
    split = lambda x: torch.stack(x.split(n))
    guess = torch.stack([split(res.snapshots[t]) for t in ts] + [split(res.x1)])
    state = torch.stack([split(res.states[t]) for t in ts] + [split(res.x1)])
    with torch.no_grad():
        s = torch.full((n,), eps, device=device)
        probs = torch.stack([F.softmax(clf(guess[-1, r].to(device), s), -1).cpu() for r in range(3)])
    torch.save({"guess": guess, "state": state, "ts": ts + [1.0], "w": args.w, "steps": args.steps,
                "p_class": torch.stack([probs[0][:, cat], probs[1][:, dog], probs[2][:, dog]]),
                "centered": config.data.centered}, f"{args.out}/frames.pt")


def sprite(frames, centered):
    """frames: (rows, F, C, H, W) -> base64 PNG with `rows` rows and F columns of tiles."""
    from PIL import Image
    x = (frames + 1) / 2 if centered else frames
    x = (x.clamp(0, 1) * 255).round().byte()
    r, f, c, h, w = x.shape
    img = x.permute(0, 3, 1, 4, 2).reshape(r * h, f * w, c).numpy()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_html(args):
    d = torch.load(f"{args.out}/frames.pt", weights_only=False)
    guess, state = d["guess"], d["state"]          # (F, 3, n, C, H, W)
    nF, _, n = guess.shape[:3]
    sq = lambda a, b: (a - b).flatten(1).pow(2).sum(1)
    paths = []
    for i in range(n):
        rows = torch.cat([guess[:, :, i].transpose(0, 1), state[:, :, i].transpose(0, 1)])  # (6, F, C, H, W)
        paths.append({
            "sprite": sprite(rows, d["centered"]),
            "same": [round(v, 1) for v in sq(guess[:, 0, i], guess[:, 1, i]).tolist()],
            "rev": [round(v, 1) for v in sq(guess[:, 0, i], guess[:, 2, i]).tolist()],
            "p": [round(v, 2) for v in d["p_class"][:, i].tolist()],
        })
    best = int(d["p_class"].min(0).values.argmax())
    data = {"ts": d["ts"], "tile": guess.shape[-1], "runs": RUNS, "paths": paths, "best": best,
            "w": d["w"], "steps": d["steps"]}
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(f"{args.out}/viewer.html", "w") as f:
        f.write(html)
    print(f"wrote {args.out}/viewer.html  ({len(html) / 1e6:.1f} MB, {n} paths × {nF} frames)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Föllmer path viewer</title>
<style>
  :root { --bg:#fbfaf7; --panel:#ffffff; --ink:#1d1c1a; --muted:#6b6862; --line:#dedad2; --track:#e9e5dc;
          --same:#2f6fd0; --rev:#d2691e; --accent:#1d1c1a; }
  @media (prefers-color-scheme: dark) { :root { --bg:#151514; --panel:#1e1e1c; --ink:#eceae4; --muted:#a09c93;
          --line:#33322f; --track:#2a2927; --same:#6aa5ff; --rev:#f0914a; --accent:#eceae4; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  main { max-width: 980px; margin: 0 auto; padding: 20px 16px 40px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  p.sub { margin: 0 0 16px; color: var(--muted); }
  .bar { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; margin-bottom:14px; }
  .seg { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .seg button { border:0; background:var(--panel); color:var(--ink); padding:6px 11px; font:inherit; cursor:pointer; border-right:1px solid var(--line); }
  .seg button:last-child { border-right:0; }
  .seg button[aria-pressed="true"] { background:var(--accent); color:var(--bg); }
  .label { color:var(--muted); font-size:13px; margin-right:6px; }
  .grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }
  figure { margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; }
  figure canvas { width:100%; aspect-ratio:1; display:block; image-rendering:pixelated; border-radius:6px; background:#000; }
  figcaption { margin-top:8px; font-size:13.5px; }
  figcaption .p { color:var(--muted); font-size:12.5px; }
  .use { margin-top:8px; }
  .use .t { height:8px; border-radius:4px; background:var(--track); position:relative; overflow:hidden; }
  .use .f { position:absolute; top:0; bottom:0; border-radius:4px; }
  .use small { color:var(--muted); font-size:12px; }
  .time { display:flex; align-items:center; gap:12px; margin:16px 0 6px; }
  .time input[type=range] { flex:1; accent-color: var(--accent); height:28px; }
  .time button { border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:8px; padding:6px 14px; font:inherit; cursor:pointer; min-width:70px; }
  .time output { font-variant-numeric: tabular-nums; min-width:74px; text-align:right; }
  .chart { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; margin-top:12px; }
  .chart h2 { font-size:14px; margin:0 0 2px; }
  .chart .legend { font-size:13px; color:var(--muted); margin-bottom:6px; display:flex; flex-wrap:wrap; gap:4px 18px; }
  .chart .legend b { font-variant-numeric: tabular-nums; color:var(--ink); }
  .sw { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:6px; border-radius:2px; }
  #chart { width:100%; height:190px; display:block; }
  .note { color:var(--muted); font-size:13px; margin-top:14px; }
  @media (max-width: 640px) { .grid { grid-template-columns:1fr; } figure { display:grid; grid-template-columns:120px 1fr; gap:10px; align-items:center; }
    figcaption { margin-top:0; } }
</style>
</head>
<body>
<main>
  <h1>One Brownian path, three processes</h1>
  <p class="sub">Drag the slider to move through time t. Each panel is a Föllmer process dX = v dt + dB, started at 0. The left two use the same path B; the right one uses the time reversal B̂<sub>t</sub> = B<sub>1</sub> − B<sub>1−t</sub>.</p>

  <div class="bar">
    <div><span class="label">Path</span><span class="seg" id="paths"></span></div>
    <div><span class="label">Show</span><span class="seg" id="modes">
      <button data-m="0" aria-pressed="true">Model's guess of the final image</button><button data-m="1" aria-pressed="false">Raw state X<sub>t</sub></button></span></div>
  </div>

  <div class="grid" id="grid"></div>

  <div class="time">
    <button id="play">Play</button>
    <input id="t" type="range" min="0" step="1" value="0" aria-label="time t">
    <output id="tout">t = 0.00</output>
  </div>

  <div class="chart">
    <h2>How far apart are the cat and the dog?</h2>
    <div class="legend">
      <span><span class="sw" style="background:var(--same)"></span>same path B: <b id="vs"></b></span>
      <span><span class="sw" style="background:var(--rev)"></span>reversed path B̂: <b id="vr"></b></span>
      <span>squared distance between the two guesses, |x̂<sub>cat</sub> − x̂<sub>dog</sub>|²</span>
    </div>
    <canvas id="chart"></canvas>
  </div>

  <p class="note" id="note"></p>
</main>

<script>
const D = __DATA__;
const F = D.ts.length, S = D.tile;
let path = D.best, mode = 0, f = F - 1, timer = null;
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const imgs = D.paths.map(p => { const im = new Image(); im.onload = draw; im.src = p.sprite; return im; });

// panels
const grid = $("grid"), ctxs = [], probs = [], fills = [];
D.runs.forEach((name, r) => {
  const fig = document.createElement("figure");
  const c = document.createElement("canvas"); c.width = c.height = S * 8;
  const cap = document.createElement("figcaption");
  const color = r === 2 ? "var(--rev)" : "var(--same)";
  cap.innerHTML = `<div><b>${name}</b></div><div class="p"></div>
    <div class="use"><div class="t"><div class="f" style="background:${color}"></div></div><small></small></div>`;
  fig.append(c, cap); grid.append(fig);
  const ctx = c.getContext("2d"); ctx.imageSmoothingEnabled = false; ctxs.push(ctx);
  probs.push(cap.querySelector(".p")); fills.push([cap.querySelector(".f"), cap.querySelector("small")]);
});

// controls
D.paths.forEach((_, i) => {
  const b = document.createElement("button"); b.textContent = i + 1 + (i === D.best ? " ★" : "");
  b.onclick = () => { path = i; sync(); }; $("paths").append(b);
});
$("modes").querySelectorAll("button").forEach(b => b.onclick = () => { mode = +b.dataset.m; sync(); });
const slider = $("t"); slider.max = F - 1; slider.value = f;
slider.oninput = () => { f = +slider.value; draw(); };
$("play").onclick = () => {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; return; }
  if (f >= F - 1) f = 0;
  $("play").textContent = "Pause";
  timer = setInterval(() => { f++; if (f >= F - 1) { f = F - 1; clearInterval(timer); timer = null; $("play").textContent = "Play"; }
    slider.value = f; draw(); }, 70);
};

function sync() {
  [...$("paths").children].forEach((b, i) => b.setAttribute("aria-pressed", i === path));
  $("modes").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", +b.dataset.m === mode));
  const p = D.paths[path].p;
  probs.forEach((el, r) => el.textContent = `classifier on final image: ${Math.round(p[r] * 100)}% ${r === 0 ? "cat" : "dog"}`);
  draw();
}

function draw() {
  const t = D.ts[f], im = imgs[path];
  $("tout").textContent = "t = " + t.toFixed(2);
  if (im.complete) ctxs.forEach((ctx, r) => ctx.drawImage(im, f * S, (mode * 3 + r) * S, S, S, 0, 0, ctx.canvas.width, ctx.canvas.height));
  fills.forEach(([bar, txt], r) => {
    const rev = r === 2;
    bar.style.left = (rev ? (1 - t) * 100 : 0) + "%"; bar.style.width = t * 100 + "%";
    txt.textContent = rev ? `has used B on [${(1 - t).toFixed(2)}, 1], read backwards` : `has used B on [0, ${t.toFixed(2)}]`;
  });
  const P = D.paths[path];
  $("vs").textContent = Math.round(P.same[f]); $("vr").textContent = Math.round(P.rev[f]);
  chart();
}

function chart() {
  const c = $("chart"), dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const P = D.paths[path], L = 46, R = 12, T = 8, Bm = 24;
  const ymax = Math.max(...P.same, ...P.rev) * 1.08 || 1;
  const X = t => L + t * (W - L - R), Y = v => T + (1 - v / ymax) * (H - T - Bm);
  g.font = "12px -apple-system, sans-serif"; g.fillStyle = css("--muted"); g.strokeStyle = css("--line"); g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const v = ymax / 1.08 * i / 4, y = Y(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.textAlign = "right"; g.fillText(Math.round(v), L - 6, y + 4); }
  g.textAlign = "center"; [0, .25, .5, .75, 1].forEach(t => g.fillText(t === 0 ? "t = 0" : t, X(t), H - 6));
  [["same", "--same"], ["rev", "--rev"]].forEach(([k, col]) => {
    g.strokeStyle = css(col); g.lineWidth = 2; g.beginPath();
    P[k].forEach((v, i) => i ? g.lineTo(X(D.ts[i]), Y(v)) : g.moveTo(X(D.ts[i]), Y(v))); g.stroke();
    g.fillStyle = css(col); g.beginPath(); g.arc(X(D.ts[f]), Y(P[k][f]), 4, 0, 7); g.fill();
  });
  g.strokeStyle = css("--muted"); g.lineWidth = 1; g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(X(D.ts[f]), T); g.lineTo(X(D.ts[f]), H - Bm); g.stroke(); g.setLineDash([]);
}

$("note").innerHTML = `${D.paths.length} Brownian paths, ${D.steps} Euler steps, classifier guidance w = ${D.w}.
  ★ marks the path whose three final images the classifier rated most clearly cat / dog / dog.
  "Model's guess" is E[X<sub>1</sub> | X<sub>t</sub>]; "Raw state" is X<sub>t</sub> itself, which starts at 0 (grey) and is noisy until late.
  At w = ${D.w} the targets are sharpened versions of the CIFAR-10 cat and dog distributions.`;
window.addEventListener("resize", chart);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", chart);
sync();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="score_sde_pytorch")
    p.add_argument("--config", default="vp/cifar10_ddpmpp_continuous")
    p.add_argument("--ckpt")
    p.add_argument("--classifier")
    p.add_argument("--w", type=float, default=6.0)
    p.add_argument("--n", type=int, default=8, help="Brownian paths")
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--out", default="results/viewer")
    p.add_argument("--html-only", action="store_true", help="rebuild the HTML from saved frames.pt")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.html_only:
        sample(args)
    build_html(args)


if __name__ == "__main__":
    main()
