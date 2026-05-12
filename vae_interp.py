import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from train_vae import VAE


def load_vae(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = VAE(cfg["flat_dim"], cfg["latent_dim"], cfg["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = torch.from_numpy(ckpt["norm"]["mean"]).to(device)
    std = torch.from_numpy(ckpt["norm"]["std"]).to(device)
    return model, cfg, mean, std


def decode(model: VAE, z: torch.Tensor, cfg: dict, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """z: (B, latent_dim) -> (B, L, pos_dim) un-normalized delta-position trajectories."""
    with torch.no_grad():
        x_hat = model.dec(z)
    L, F = cfg["window"], cfg["pos_dim"]
    x_hat = x_hat.view(-1, L, F)
    x_hat = x_hat * std.view(1, 1, F) + mean.view(1, 1, F)
    return x_hat.cpu().numpy()


def interpolation(model, cfg, mean, std, latent_dim, latent_seed, scale, num, device):
    g = torch.Generator(device=device).manual_seed(latent_seed)
    z_a = torch.randn(latent_dim, generator=g, device=device) * scale
    z_b = torch.randn(latent_dim, generator=g, device=device) * scale
    alphas = torch.linspace(0.0, 1.0, num, device=device).unsqueeze(1)
    zs = (1.0 - alphas) * z_a.unsqueeze(0) + alphas * z_b.unsqueeze(0)
    return decode(model, zs, cfg, mean, std)


def main():
    p = argparse.ArgumentParser(description="Sample/interpolate from a trained delta-position-trajectory VAE and plot the decoded trajectories.")
    p.add_argument("--ckpt", type=str, default="vae.pt")
    p.add_argument("--out", type=str, default="vae_interp.png")
    p.add_argument("--rows", type=int, default=5, help="Number of independent interpolations to display.")
    p.add_argument("--num", type=int, default=9, help="Number of points per interpolation.")
    p.add_argument("--latent_seed", type=int, default=42, help="Base seed; row i uses latent_seed + i.")
    p.add_argument("--scale", type=float, default=2.5, help="Std multiplier when sampling endpoints.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, cfg, mean, std = load_vae(args.ckpt, args.device)
    print(f"Loaded VAE: window={cfg['window']} pos_dim={cfg['pos_dim']} latent_dim={cfg['latent_dim']}")

    F = cfg["pos_dim"]
    L = cfg["window"]
    t = np.arange(L)
    cmap = plt.get_cmap("viridis")
    dim_names = ["Δx", "Δy", "Δyaw"][:F] + [f"d{i}" for i in range(3, F)]

    fig = plt.figure(figsize=(13, 2.6 * args.rows))
    outer = GridSpec(args.rows, 2, figure=fig, width_ratios=[1.0, 1.6], wspace=0.15, hspace=0.45)

    for r in range(args.rows):
        trajs = interpolation(model, cfg, mean, std, cfg["latent_dim"],
                              latent_seed=args.latent_seed + r,
                              scale=args.scale, num=args.num, device=args.device)
        # trajs: (num, L, pos_dim) — already in delta-position units.

        ax_xy = fig.add_subplot(outer[r, 0])
        for i in range(args.num):
            c = cmap(i / max(args.num - 1, 1))
            path = trajs[i]
            ax_xy.plot(path[:, 0], path[:, 1] if F > 1 else np.zeros_like(path[:, 0]),
                       color=c, lw=1.3, alpha=0.8,
                       marker="o", markersize=2.5, markerfacecolor=c, markeredgecolor="none")
        ax_xy.set_aspect("equal")
        ax_xy.set_xticks([]); ax_xy.set_yticks([])
        ax_xy.set_ylabel(f"interp #{r+1}", fontsize=10)

        inner = GridSpecFromSubplotSpec(F, 1, subplot_spec=outer[r, 1], hspace=0.25)
        for d in range(F):
            sub = fig.add_subplot(inner[d, 0])
            for i in range(args.num):
                c = cmap(i / max(args.num - 1, 1))
                sub.plot(t, trajs[i, :, d], color=c, lw=1.2, alpha=0.9)
            sub.grid(alpha=0.25)
            sub.tick_params(labelsize=8)
            if r == 0:
                sub.set_title(dim_names[d], fontsize=9)
            if d < F - 1:
                sub.set_xticklabels([])
            else:
                sub.set_xlabel("step", fontsize=9)

    fig.suptitle(f"VAE latent interpolations (rows={args.rows}, num={args.num}, scale={args.scale})", fontsize=13)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
