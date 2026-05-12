import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path):
    """Read eval_many output and pull the seed out of the ckpt filename (bc_<arch>_d<N>_s<S>.pt)."""
    seed_re = re.compile(r"_s(\d+)\.pt$")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            m = seed_re.search(r["ckpt"])
            if not m:
                continue
            rows.append({
                "arch": r["arch"],
                "demos": int(r["demos"]),
                "seed": int(m.group(1)),
                "rate": float(r["rate"]),
                "successes": int(r["successes"]),
                "episodes": int(r["episodes"]),
            })
    return rows


def plot(rows, out_path, title_suffix=""):
    by_arch_demos = defaultdict(list)
    for r in rows:
        by_arch_demos[(r["arch"], r["demos"])].append(r["rate"])

    arches = []
    for r in rows:
        if r["arch"] not in arches:
            arches.append(r["arch"])
    demo_counts = sorted({r["demos"] for r in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("tab10").colors
    for ai, arch in enumerate(arches):
        c = colors[ai % len(colors)]
        xs, means, stds = [], [], []
        for d in demo_counts:
            cell = by_arch_demos.get((arch, d), [])
            if not cell:
                continue
            xs.append(d)
            means.append(100 * np.mean(cell))
            stds.append(100 * np.std(cell, ddof=1) if len(cell) > 1 else 0.0)
            # per-seed scatter (jittered slightly so dots from different arches don't overlap)
            jitter = 1.0 + 0.04 * (ai - (len(arches) - 1) / 2)
            ax.scatter([d * jitter] * len(cell), [100 * v for v in cell],
                       color=c, alpha=0.45, s=18, zorder=2)
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", color=c, label=f"{arch} (mean ± seed σ)",
                        linewidth=2, markersize=7, capsize=4, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(left=10)
    ax.set_xlabel("# demonstration episodes")
    ax.set_ylabel("Success rate (%)")
    n_seeds = max((len(v) for v in by_arch_demos.values()), default=0)
    n_ep = rows[0]["episodes"] if rows else 0
    ax.set_title(f"Seed variability ({n_seeds} seeds × {n_ep} eval episodes){title_suffix}")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot seed-variability eval results: mean ± std across seeds, per arch, vs demo count.")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--title_suffix", type=str, default="")
    args = p.parse_args()
    out = args.out or os.path.splitext(args.csv)[0] + ".png"
    plot(load_rows(args.csv), out, title_suffix=args.title_suffix)


if __name__ == "__main__":
    main()
