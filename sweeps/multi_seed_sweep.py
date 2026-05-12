import argparse
import csv
import os
import queue
import re
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

# Sibling entry-point scripts live at the repo root, one level up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BC = os.path.join(REPO_ROOT, "train_bc.py")
TRAIN_VAE = os.path.join(REPO_ROOT, "train_vae.py")
EVAL_BC = os.path.join(REPO_ROOT, "eval_bc.py")

ARCHS = [("mlp", None), ("dec_z4", 4), ("dec_z8", 8), ("dec_z16", 16)]
DEMO_COUNTS = [25, 50, 100, 250, 500, 1000]


def _run(cmd, gpu, label):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[gpu{gpu}] start: {label}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"!! [gpu{gpu}] {label} FAILED rc={r.returncode}\n{r.stderr[-800:]}", flush=True)
        return r
    last = r.stdout.strip().splitlines()[-2:]
    print(f"[gpu{gpu}] done: {label}  ::  {' | '.join(last)}", flush=True)
    return r


def parallel_run(jobs, slots):
    """jobs: list of (cmd, label). slots: list of GPU ids; concurrency is len(slots)."""
    if not jobs:
        return
    gpu_q = queue.Queue()
    for g in slots:
        gpu_q.put(g)

    def task(cmd, label):
        gpu = gpu_q.get()
        try:
            return _run(cmd, gpu, label)
        finally:
            gpu_q.put(gpu)

    with ThreadPoolExecutor(max_workers=len(slots)) as ex:
        futures = [ex.submit(task, c, l) for c, l in jobs]
        for f in futures:
            f.result()


def train_vaes(args, archs):
    if args.fixed_vae:
        print(f"\n[phase 1] skipping VAE training — using fixed VAE {args.fixed_vae} for all latent BC runs")
        return
    jobs = []
    for _arch_name, z in archs:
        if z is None:
            continue
        for seed in args.seeds:
            path = os.path.join(args.out_dir, f"vae_z{z}_s{seed}.pt")
            if os.path.exists(path):
                print(f"[vae] reusing {path}")
                continue
            cmd = [
                sys.executable, TRAIN_VAE,
                "--out", path,
                "--latent_dim", str(z),
                "--window", str(args.window),
                "--pos_dim", "2",
                "--beta", str(args.beta),
                "--max_steps", str(args.vae_steps),
                "--seed", str(seed),
                "--wandb_project", args.wandb_project,
            ]
            jobs.append((cmd, f"vae z={z} s={seed}"))
    print(f"\n[phase 1] {len(jobs)} VAE jobs across GPUs {args.gpus}")
    parallel_run(jobs, args.gpus)


def train_bcs(args, archs):
    jobs = []
    for arch_name, z in archs:
        for seed in args.seeds:
            if z is None:
                vae_p = None
            elif args.fixed_vae:
                vae_p = args.fixed_vae
            else:
                vae_p = os.path.join(args.out_dir, f"vae_z{z}_s{seed}.pt")
            for demos in args.demos:
                tag = f"{arch_name}_d{demos}_s{seed}"
                ckpt = os.path.join(args.out_dir, f"bc_{tag}.pt")
                if os.path.exists(ckpt):
                    print(f"[bc] reusing {ckpt}")
                    continue
                cmd = [
                    sys.executable, TRAIN_BC,
                    "--out", ckpt,
                    "--log_csv", os.path.join(args.out_dir, f"bc_{tag}.csv"),
                    "--max_episodes", str(demos),
                    "--max_steps", str(args.max_steps),
                    "--eval_every", str(args.max_steps + 1),   # disable in-loop eval
                    "--eval_episodes", "0",
                    "--window", str(args.window),
                    "--seed", str(seed),
                    "--tag", tag,
                    "--wandb_project", args.wandb_project,
                ]
                if vae_p is not None:
                    cmd += ["--vae_ckpt", vae_p]
                jobs.append((cmd, tag))
    print(f"\n[phase 2] {len(jobs)} BC jobs across GPUs {args.gpus}")
    parallel_run(jobs, args.gpus)


SUCCESS_RE = re.compile(r"Success:\s+(\d+)/(\d+)")


def eval_all(args, archs):
    ckpts = []
    for arch_name, _z in archs:
        for seed in args.seeds:
            for demos in args.demos:
                p = os.path.join(args.out_dir, f"bc_{arch_name}_d{demos}_s{seed}.pt")
                if os.path.exists(p):
                    ckpts.append((arch_name, demos, seed, p))

    # Run `args.eval_per_gpu` concurrent eval_bc.py instances per GPU (eval is CPU-bound).
    slots = [g for g in args.gpus for _ in range(args.eval_per_gpu)]
    gpu_q = queue.Queue()
    for g in slots:
        gpu_q.put(g)

    rows = []
    rows_lock = threading.Lock()

    def task(arch_name, demos, seed, ckpt):
        gpu = gpu_q.get()
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [sys.executable, EVAL_BC, "--ckpt", ckpt, "--episodes", str(args.eval_episodes_final)]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if r.returncode != 0:
                print(f"!! [gpu{gpu}] eval FAILED {ckpt} rc={r.returncode}", flush=True)
                return
            m = SUCCESS_RE.search(r.stdout)
            if not m:
                print(f"!! couldn't parse eval output for {ckpt}", flush=True)
                return
            s, n = int(m.group(1)), int(m.group(2))
            row = {
                "ckpt": os.path.basename(ckpt), "arch": arch_name, "demos": demos, "seed": seed,
                "successes": s, "episodes": n, "rate": s / max(n, 1),
            }
            with rows_lock:
                rows.append(row)
            print(f"[gpu{gpu}] eval {arch_name}_d{demos}_s{seed}: {s}/{n} ({100*s/n:.1f}%)", flush=True)
        finally:
            gpu_q.put(gpu)

    with ThreadPoolExecutor(max_workers=len(slots)) as ex:
        futures = [ex.submit(task, *t) for t in ckpts]
        for f in futures:
            f.result()
    return rows


def write_csv(rows, path):
    rows = sorted(rows, key=lambda r: (r["arch"], r["demos"], r["seed"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "arch", "demos", "seed", "successes", "episodes", "rate"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved {path} ({len(rows)} rows)")


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "ckpt": r["ckpt"], "arch": r["arch"],
                "demos": int(r["demos"]), "seed": int(r["seed"]),
                "successes": int(r["successes"]), "episodes": int(r["episodes"]),
                "rate": float(r["rate"]),
            })
    return rows


def plot(rows, out_path, title):
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["arch"], r["demos"])].append(r["rate"])

    arch_order = [a for a, _ in ARCHS]
    arches = [a for a in arch_order if any(r["arch"] == a for r in rows)]
    demo_counts = sorted({r["demos"] for r in rows})

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = plt.get_cmap("tab10").colors
    for ai, arch in enumerate(arches):
        c = colors[ai % len(colors)]
        xs, means, stds = [], [], []
        for d in demo_counts:
            cell = by_cell.get((arch, d), [])
            if not cell:
                continue
            xs.append(d)
            means.append(100 * np.mean(cell))
            stds.append(100 * np.std(cell, ddof=1) if len(cell) > 1 else 0.0)
            jit = 1.0 + 0.04 * (ai - (len(arches) - 1) / 2)
            ax.scatter([d * jit] * len(cell), [100 * v for v in cell],
                       color=c, alpha=0.4, s=15, zorder=2)
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", color=c, label=arch,
                        linewidth=2, markersize=6, capsize=4, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(left=10)
    ax.set_ylim(0, 105)
    ax.set_xlabel("# demonstration episodes")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Multi-seed (β=0) sample-efficiency sweep: 4 archs × N demo counts × M seeds. Trains in parallel across GPUs, evals each ckpt at a fixed episode count, plots mean ± seed σ.")
    p.add_argument("--out_dir", type=str, default="sweep_multi")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--demos", type=int, nargs="+", default=DEMO_COUNTS)
    p.add_argument("--archs", type=str, default=None,
                   help="Comma-separated arch subset (default: all four).")
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--vae_steps", type=int, default=20000)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--eval_per_gpu", type=int, default=2,
                   help="Concurrent eval_bc.py workers per GPU (eval is CPU-bound, so >1 helps).")
    p.add_argument("--eval_episodes_final", type=int, default=100)
    p.add_argument("--wandb_project", type=str, default="onceler-multiseed-beta0")
    p.add_argument("--fixed_vae", type=str, default=None,
                   help="Use this VAE ckpt for all latent BC runs (skips VAE training). For isolating BC-seed variance.")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true",
                   help="Only plot (requires eval CSV already exists).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    arch_filter = set(args.archs.split(",")) if args.archs else None
    archs = [(n, z) for (n, z) in ARCHS if arch_filter is None or n in arch_filter]

    if not args.skip_train:
        train_vaes(args, archs)
        train_bcs(args, archs)

    eval_csv = os.path.join(args.out_dir, f"eval_{args.eval_episodes_final}.csv")
    if args.skip_eval:
        rows = load_csv(eval_csv)
    else:
        rows = eval_all(args, archs)
        write_csv(rows, eval_csv)

    plot_path = os.path.join(args.out_dir, f"eval_{args.eval_episodes_final}.png")
    plot(rows, plot_path,
         title=f"BC sample efficiency, β={args.beta} ({len(args.seeds)} seeds × {args.eval_episodes_final} eval episodes)")


if __name__ == "__main__":
    main()
