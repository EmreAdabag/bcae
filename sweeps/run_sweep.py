import argparse
import csv
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

# Sibling entry-point scripts live at the repo root, one level up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BC = os.path.join(REPO_ROOT, "train_bc.py")
TRAIN_VAE = os.path.join(REPO_ROOT, "train_vae.py")


# (arch_name, latent_dim or None for the no-VAE baseline)
ARCHS = [
    ("mlp", None),
    ("dec_z4", 4),
    ("dec_z8", 8),
    ("dec_z16", 16),
]

DEMO_COUNTS = [25, 50, 100, 250, 500, 1000]


def _run(cmd, label):
    print(f"\n=== {label} ===")
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"!! {label} FAILED (rc={r.returncode})")
        print(r.stderr[-2000:])
        return None
    last_lines = r.stdout.strip().split("\n")[-6:]
    for line in last_lines:
        print("  " + line)
    return r


def ensure_vae(z, out_dir, args):
    """Train a VAE at latent_dim=z into out_dir if no checkpoint is present.
    pos_dim is pinned to 2 to match BC's (dx, dy) action space — a 3D-feature VAE would
    silently change BC's pos_dim via the saved vae_cfg."""
    path = os.path.join(out_dir, f"vae_z{z}.pt")
    if os.path.exists(path):
        print(f"[vae] reusing {path}")
        return path
    cmd = [
        sys.executable, TRAIN_VAE,
        "--out", path,
        "--latent_dim", str(z),
        "--window", str(args.window),
        "--pos_dim", "2",
        "--beta", str(args.beta),
        "--max_steps", str(args.vae_steps),
        "--wandb_project", f"onceler-vae{args.wandb_suffix}",
    ]
    return path if _run(cmd, f"train vae z={z}") is not None else None


def run_one(arch_name, vae_path, demos, args, out_dir):
    tag = f"{arch_name}_d{demos}"
    ckpt_path = os.path.join(out_dir, f"bc_{tag}.pt")
    log_path = os.path.join(out_dir, f"bc_{tag}.csv")
    viz_dir = os.path.join(out_dir, f"viz_{tag}")
    cmd = [
        sys.executable, TRAIN_BC,
        "--out", ckpt_path,
        "--log_csv", log_path,
        "--max_episodes", str(demos),
        "--max_steps", str(args.max_steps),
        "--eval_every", str(args.eval_every),
        "--eval_episodes", str(args.eval_episodes),
        "--window", str(args.window),
        "--viz_dir", viz_dir,
        "--tag", tag,
        "--wandb_project", f"onceler-bc{args.wandb_suffix}",
    ]
    if vae_path is not None:
        cmd += ["--vae_ckpt", vae_path]

    if _run(cmd, tag) is None:
        return []

    rows = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for row in csv.DictReader(f):
                rows.append({
                    "arch": arch_name,
                    "demos": demos,
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "success": float(row["success"]),
                })
    return rows


def plot_results(all_rows, out_dir, max_steps):
    if not all_rows:
        print("No rows to plot.")
        return

    arches = [a for a, _ in ARCHS if any(r["arch"] == a for r in all_rows)]

    # Plot 1: final-step success vs # demos, one line per arch.
    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in arches:
        xs, ys = [], []
        for d in DEMO_COUNTS:
            rows = [r for r in all_rows if r["arch"] == arch and r["demos"] == d]
            if not rows:
                continue
            final = max(rows, key=lambda r: r["step"])
            xs.append(d); ys.append(final["success"] * 100)
        if xs:
            ax.plot(xs, ys, marker="o", label=arch, linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("# demonstration episodes")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"BC sample efficiency: final-step success ({max_steps} steps)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    p = os.path.join(out_dir, "summary_demos.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")

    # Plot 2: learning curves (success vs step), 1 subplot per arch, line per demo count.
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(arches), figsize=(4 * len(arches), 4), sharey=True)
    if len(arches) == 1:
        axes = [axes]
    for ai, arch in enumerate(arches):
        ax = axes[ai]
        for di, d in enumerate(DEMO_COUNTS):
            rows = sorted([r for r in all_rows if r["arch"] == arch and r["demos"] == d],
                          key=lambda r: r["step"])
            if not rows:
                continue
            xs = [r["step"] for r in rows]
            ys = [r["success"] * 100 for r in rows]
            color = cmap(di / max(len(DEMO_COUNTS) - 1, 1))
            ax.plot(xs, ys, marker="o", color=color, label=f"{d}", linewidth=1.5, markersize=4)
        ax.set_title(arch)
        ax.set_xlabel("training step")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        if ai == 0:
            ax.set_ylabel("Success rate (%)")
        if ai == len(arches) - 1:
            ax.legend(title="demos", fontsize=8, loc="lower right")
    fig.suptitle(f"BC learning curves across data mixtures ({max_steps} steps)")
    p = os.path.join(out_dir, "summary_curves.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")


def main():
    p = argparse.ArgumentParser(description="Run a 4-arch x N-demo BC sample-efficiency sweep and plot the results.")
    p.add_argument("--out_dir", type=str, default="sweep")
    p.add_argument("--max_steps", type=int, default=10000,
                   help="BC training budget in gradient steps (shared across demo counts for fair compute).")
    p.add_argument("--eval_every", type=int, default=5000,
                   help="Run env eval (and write viz) every N BC training steps. Default gives 2 evals per run (mid + end).")
    p.add_argument("--eval_episodes", type=int, default=30)
    p.add_argument("--window", type=int, default=16,
                   help="Action-chunk length, passed to both train_vae and train_bc so they stay in sync.")
    p.add_argument("--vae_steps", type=int, default=20000,
                   help="VAE pretraining budget in gradient steps.")
    p.add_argument("--beta", type=float, default=0.0,
                   help="VAE KL weight. 0 = deterministic AE; 1 = standard VAE.")
    p.add_argument("--wandb_suffix", type=str, default="",
                   help="Appended to wandb project names (e.g. '-beta1') so parallel sweeps stay separate.")
    p.add_argument("--archs", type=str, default=None,
                   help="Comma-separated subset of arch names to run (default: all).")
    p.add_argument("--demos", type=str, default=None,
                   help="Comma-separated subset of demo counts to run (default: all).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    arch_filter = set(args.archs.split(",")) if args.archs else None
    demos_filter = set(int(d) for d in args.demos.split(",")) if args.demos else None
    archs = [(n, z) for (n, z) in ARCHS if arch_filter is None or n in arch_filter]
    demo_counts = [d for d in DEMO_COUNTS if demos_filter is None or d in demos_filter]

    # Train each needed VAE up front so every BC demo-count run for the same arch shares
    # one decoder (and so latent_dim is held fixed across the demo-count axis).
    arch_vae_paths = {n: (None if z is None else ensure_vae(z, args.out_dir, args)) for n, z in archs}

    all_rows = []
    for arch_name, _z in archs:
        vae_path = arch_vae_paths[arch_name]
        for demos in demo_counts:
            all_rows.extend(run_one(arch_name, vae_path, demos, args, args.out_dir))

    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "demos", "epoch", "step", "success"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nSaved {summary_csv} with {len(all_rows)} rows")

    plot_results(all_rows, args.out_dir, args.max_steps)


if __name__ == "__main__":
    main()
