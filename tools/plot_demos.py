import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Repo root holds env.py and train_vae.py; tools/ holds vae_interp.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import PickAndPlaceEnv
from vae_interp import rollout
from train_vae import make_windows


def main():
    p = argparse.ArgumentParser(description="Plot ground-truth training demo windows in the same layout as vae_interp.py.")
    p.add_argument("--data", type=str, default="dataset.npz")
    p.add_argument("--out", type=str, default="demos.png")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--rows", type=int, default=5, help="Number of demos to display (one per row).")
    p.add_argument("--env_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling demo windows.")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    data = np.load(args.data)
    act, ep = data["act"], data["episode"]
    A = act.shape[1]
    L = args.window
    print(f"Loaded {len(act)} transitions, act_dim={A}")

    windows = make_windows(act, ep, L)
    print(f"Built {len(windows)} windows of shape {windows.shape[1:]}")

    rng = np.random.default_rng(args.seed)
    if args.rows > len(windows):
        raise ValueError(f"Need {args.rows} windows but dataset only has {len(windows)}")
    idx = rng.choice(len(windows), size=args.rows, replace=False)
    selected = windows[idx]  # (rows, L, A)

    t = np.arange(L)
    cmap = plt.get_cmap("viridis")
    dim_names = (["thrust", "yaw_rate", "gripper"] + [f"a{i}" for i in range(3, A)])[:A]

    env = PickAndPlaceEnv()

    fig = plt.figure(figsize=(13, 2.6 * args.rows))
    outer = GridSpec(args.rows, 2, figure=fig, width_ratios=[1.0, 1.6], wspace=0.15, hspace=0.45)

    for r in range(args.rows):
        actions = selected[r]  # (L, A)
        path = rollout(env, actions, seed=args.env_seed)
        c = cmap(r / max(args.rows - 1, 1))

        ax_xy = fig.add_subplot(outer[r, 0])
        ax_xy.plot(path[:, 0], path[:, 1], color=c, lw=1.3, alpha=0.9,
                   marker="o", markersize=2.5, markerfacecolor=c, markeredgecolor="none")
        ax_xy.set_aspect("equal")
        ax_xy.set_xticks([]); ax_xy.set_yticks([])
        ax_xy.set_ylabel(f"demo #{r+1}", fontsize=10)

        inner = GridSpecFromSubplotSpec(A, 1, subplot_spec=outer[r, 1], hspace=0.25)
        for d in range(A):
            sub = fig.add_subplot(inner[d, 0])
            sub.plot(t, actions[:, d], color=c, lw=1.2, alpha=0.9)
            sub.grid(alpha=0.25)
            sub.tick_params(labelsize=8)
            if r == 0:
                sub.set_title(dim_names[d], fontsize=9)
            if d < A - 1:
                sub.set_xticklabels([])
            else:
                sub.set_xlabel("step", fontsize=9)

    fig.suptitle(f"Training demo windows (rows={args.rows})", fontsize=13)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
