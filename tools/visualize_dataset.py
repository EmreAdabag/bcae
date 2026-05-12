import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(
        description="Render every (x, y, yaw) observation from a collected dataset."
    )
    p.add_argument("--data", type=str, default="dataset.npz")
    p.add_argument("--out", type=str, default="dataset_viz.png")
    p.add_argument("--arrow-length", type=float, default=0.02,
                   help="Length of the yaw arrow in world units.")
    p.add_argument("--stride", type=int, default=1,
                   help="Plot every Nth observation (1 = all).")
    p.add_argument("--episodes", type=int, default=None,
                   help="Plot only the first N episodes (default: all).")
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    data = np.load(args.data)
    obs, ep = data["obs"], data["episode"]
    total_ep = int(ep.max()) + 1
    if args.episodes is not None:
        mask = ep < args.episodes
        obs = obs[mask]
        ep = ep[mask]
    # env's _get_obs lays out (x, y, cos_yaw, sin_yaw, ...).
    x = obs[::args.stride, 0]
    y = obs[::args.stride, 1]
    yaw = np.arctan2(obs[::args.stride, 3], obs[::args.stride, 2])
    ep = ep[::args.stride]
    n_ep = int(ep.max()) + 1 if len(ep) else 0
    print(f"Loaded {len(obs)} observations across {n_ep}/{total_ep} episodes "
          f"({len(x)} shown after stride={args.stride})")

    u = np.cos(yaw) * args.arrow_length
    v = np.sin(yaw) * args.arrow_length

    cmap = plt.get_cmap("hsv")
    colors = cmap((ep % n_ep) / max(n_ep, 1))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.quiver(
        x, y, u, v,
        color=colors, alpha=args.alpha,
        angles="xy", scale_units="xy", scale=1,
        width=0.0025, headwidth=3.5, headlength=4.0,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Agent positions & yaws — {len(x)} obs, {n_ep} episodes")
    ax.grid(alpha=0.2)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
