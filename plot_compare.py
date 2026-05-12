import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from plot_eval import wilson_ci


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "arch": r["arch"],
                "demos": int(r["demos"]),
                "successes": int(r["successes"]),
                "episodes": int(r["episodes"]),
            })
    return rows


def main():
    p = argparse.ArgumentParser(description="Overlay several eval_many CSVs on one plot per arch (cell = arch × β).")
    p.add_argument("--csvs", nargs="+", required=True,
                   help="One CSV per condition; pair with --labels in matching order.")
    p.add_argument("--labels", nargs="+", required=True,
                   help="Display label for each CSV (e.g. 'β=0').")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--title", type=str, default="BC sample efficiency vs β")
    args = p.parse_args()
    assert len(args.csvs) == len(args.labels), "csvs and labels must be the same length"

    # data[arch][label] = list of (demos, rate, lo, hi) sorted by demos.
    data = defaultdict(dict)
    arches_seen = []
    for path, lbl in zip(args.csvs, args.labels):
        rows = load(path)
        by_arch = defaultdict(list)
        for r in rows:
            p_hat, p_lo, p_hi = wilson_ci(r["successes"], r["episodes"])
            by_arch[r["arch"]].append((r["demos"], p_hat * 100,
                                      (p_hat - p_lo) * 100, (p_hi - p_hat) * 100))
        for arch, pts in by_arch.items():
            pts.sort()
            data[arch][lbl] = pts
            if arch not in arches_seen:
                arches_seen.append(arch)

    fig, axes = plt.subplots(1, len(arches_seen), figsize=(4.2 * len(arches_seen), 4.2),
                             sharey=True)
    if len(arches_seen) == 1:
        axes = [axes]
    colors = plt.get_cmap("tab10").colors
    for ai, arch in enumerate(arches_seen):
        ax = axes[ai]
        for li, lbl in enumerate(args.labels):
            if lbl not in data[arch]:
                continue
            pts = data[arch][lbl]
            xs = [d for d, *_ in pts]
            ys = [r for _, r, *_ in pts]
            lo = [l for _, _, l, _ in pts]
            hi = [h for _, _, _, h in pts]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", label=lbl,
                        color=colors[li % len(colors)], linewidth=1.7,
                        markersize=5, capsize=3)
        ax.set_xscale("log")
        ax.set_xlim(left=10)
        ax.set_ylim(0, 105)
        ax.set_title(arch)
        ax.set_xlabel("# demonstration episodes")
        ax.grid(alpha=0.3, which="both")
        if ai == 0:
            ax.set_ylabel("Success rate (%)")
        if ai == len(arches_seen) - 1:
            ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(args.title)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
