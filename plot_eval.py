import argparse
import csv
import math
import os

import matplotlib.pyplot as plt


def wilson_ci(successes, episodes, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Handles rate=0 and rate=1 better
    than the normal approximation (whose σ collapses to 0 at the boundary)."""
    if episodes == 0:
        return 0.0, 0.0, 0.0
    p = successes / episodes
    n = episodes
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "arch": r["arch"],
                "demos": int(r["demos"]),
                "successes": int(r["successes"]),
                "episodes": int(r["episodes"]),
                "rate": float(r["rate"]),
            })
    return rows


def plot(rows, out_path, title_suffix=""):
    arches = []
    for r in rows:
        if r["arch"] not in arches:
            arches.append(r["arch"])
    demo_counts = sorted({r["demos"] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in arches:
        xs, ys, lo, hi = [], [], [], []
        for d in demo_counts:
            cell = [r for r in rows if r["arch"] == arch and r["demos"] == d]
            if not cell:
                continue
            r = cell[0]
            p, p_lo, p_hi = wilson_ci(r["successes"], r["episodes"])
            xs.append(d)
            ys.append(p * 100)
            lo.append((p - p_lo) * 100)
            hi.append((p_hi - p) * 100)
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", label=arch,
                        linewidth=2, markersize=6, capsize=3)
    ax.set_xscale("log")
    ax.set_xlim(left=10)
    ax.set_xlabel("# demonstration episodes")
    ax.set_ylabel("Success rate (%)")
    n_ep = rows[0]["episodes"] if rows else 0
    ax.set_title(f"BC sample efficiency ({n_ep} eval episodes){title_suffix}")
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
    p = argparse.ArgumentParser(description="Plot eval_many.py results: success rate vs demonstration count, one line per arch.")
    p.add_argument("--csv", type=str, required=True, help="CSV produced by eval_many.py.")
    p.add_argument("--out", type=str, default=None, help="Output PNG path. Defaults to <csv>.png.")
    p.add_argument("--title_suffix", type=str, default="", help="Optional suffix appended to the plot title.")
    args = p.parse_args()

    out = args.out or os.path.splitext(args.csv)[0] + ".png"
    rows = load_rows(args.csv)
    plot(rows, out, title_suffix=args.title_suffix)


if __name__ == "__main__":
    main()
