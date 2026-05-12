import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_windows(pos: np.ndarray, ep: np.ndarray, L: int) -> np.ndarray:
    """Slice length-L position windows as deltas from each window's first step.
    Window row 0 is always all zeros; row i = pos[t+i] - pos[t].
    Matches the (dx, dy, dyaw) target shape produced by train_bc.make_pairs."""
    feats = pos.astype(np.float32).copy()  # (N, feat_dim) = (x, y, yaw)
    windows = []
    starts = np.where(np.concatenate([[True], ep[1:] != ep[:-1]]))[0]
    ends = np.concatenate([starts[1:], [len(ep)]])
    for s, e in zip(starts, ends):
        feats[s:e, 2] = np.unwrap(feats[s:e, 2])  # continuous yaw within an episode
        if e - s < L:
            continue
        for i in range(s, e - L + 1):
            w = feats[i : i + L]
            windows.append(w - w[0])
    return np.stack(windows, axis=0)  # (num_windows, L, feat_dim)


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta: float):
    # Per-element MSE summed across feature dim, mean across batch.
    recon = F.mse_loss(x_hat, x, reduction="none").sum(dim=1).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    # beta=0 turns this into a deterministic AE — drop the KL term so logvar can drift
    # freely without contributing to the loss (and avoid the 0*inf risk if it diverges).
    total = recon if beta == 0.0 else recon + beta * kl
    return total, recon, kl


def main():
    p = argparse.ArgumentParser(description="Train an unconditional flatten-MLP VAE on fixed-length trajectory windows.")
    p.add_argument("--data", type=str, default="dataset.npz")
    p.add_argument("--out", type=str, default="vae.pt")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--pos_dim", type=int, default=2,
                   help="Position dims to keep from (x, y, yaw). Defaults to 2 to match BC's action space (dx, dy).")
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL weight. 0 = deterministic autoencoder (encoder collapses to a point estimate).")
    p.add_argument("--max_steps", type=int, default=20000,
                   help="Total number of gradient steps. Data loader is restarted as needed.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50, help="Log train (avg over interval) and val every N steps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-off", dest="wandb", action="store_false", help="Pass to disable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="onceler-vae",
                   help="wandb project name (use a unique suffix to keep parallel sweeps separate).")
    args = p.parse_args()

    wb = None
    if args.wandb:
        import wandb as wb
        wb.init(project=args.wandb_project, name=None, config=vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data)
    obs, ep = data["obs"], data["episode"]
    # env's _get_obs lays out (x, y, cos_yaw, sin_yaw, ...).
    pos = np.stack([obs[:, 0], obs[:, 1], np.arctan2(obs[:, 3], obs[:, 2])], axis=1).astype(np.float32)
    print(f"Loaded {len(obs)} transitions")

    # make_windows always produces (dx, dy, dyaw); slice to match BC's action space.
    # Yaw deltas have ~6x the std of xy deltas (turn-in-place spikes hit ~3 rad), so
    # on small datasets they dominate the z-scored loss and crowd out xy capacity.
    # Yaw isn't used at rollout anyway.
    windows = make_windows(pos, ep, args.window)[..., :args.pos_dim]  # (N, L, pos_dim)
    pos_dim = args.pos_dim
    feat_dim = pos_dim
    print(f"Built {len(windows)} windows of shape {windows.shape[1:]}")

    # Train/val split on windows.
    perm = np.random.permutation(len(windows))
    n_val = int(len(windows) * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_w, val_w = windows[train_idx], windows[val_idx]

    # Per-feature z-score using training stats only (broadcast over time).
    mean = train_w.reshape(-1, feat_dim).mean(axis=0)
    std = train_w.reshape(-1, feat_dim).std(axis=0) + 1e-6
    train_w = (train_w - mean) / std
    val_w = (val_w - mean) / std

    flat_dim = args.window * feat_dim
    train_x = torch.from_numpy(train_w.reshape(-1, flat_dim))
    val_x = torch.from_numpy(val_w.reshape(-1, flat_dim))

    train_loader = DataLoader(TensorDataset(train_x), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_x), batch_size=args.batch_size, shuffle=False)

    model = VAE(flat_dim, args.latent_dim, args.hidden).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_validation():
        model.eval()
        with torch.no_grad():
            vl_loss = vl_rec = vl_kl = 0.0
            for (xb,) in val_loader:
                xb = xb.to(args.device)
                x_hat, mu, logvar = model(xb)
                loss, rec, kl = vae_loss(xb, x_hat, mu, logvar, args.beta)
                vl_loss += loss.item(); vl_rec += rec.item(); vl_kl += kl.item()
            m = max(len(val_loader), 1)
            return vl_loss / m, vl_rec / m, vl_kl / m

    global_step = 0
    run_loss = run_rec = run_kl = 0.0
    run_count = 0
    steps_per_epoch = max(len(train_loader), 1)
    model.train()
    done = False
    while not done:
        for (xb,) in train_loader:
            if global_step >= args.max_steps:
                done = True
                break
            xb = xb.to(args.device)
            x_hat, mu, logvar = model(xb)
            loss, rec, kl = vae_loss(xb, x_hat, mu, logvar, args.beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item(); run_rec += rec.item(); run_kl += kl.item()
            run_count += 1
            global_step += 1

            if global_step % args.log_every == 0 or global_step == args.max_steps:
                tr_loss = run_loss / run_count
                tr_rec = run_rec / run_count
                tr_kl = run_kl / run_count
                run_loss = run_rec = run_kl = 0.0
                run_count = 0

                vl_loss, vl_rec, vl_kl = run_validation()
                model.train()

                epoch = (global_step - 1) // steps_per_epoch + 1
                print(f"step {global_step:6d} (epoch {epoch:3d})  "
                      f"train loss={tr_loss:.3f} rec={tr_rec:.3f} kl={tr_kl:.3f}   "
                      f"val loss={vl_loss:.3f} rec={vl_rec:.3f} kl={vl_kl:.3f}")

                if wb is not None:
                    wb.log({
                        "epoch": epoch,
                        "train/loss": tr_loss, "train/recon": tr_rec, "train/kl": tr_kl,
                        "val/loss": vl_loss, "val/recon": vl_rec, "val/kl": vl_kl,
                    }, step=global_step)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "window": args.window,
            "pos_dim": pos_dim,
            "feat_dim": feat_dim,
            "flat_dim": flat_dim,
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
        },
        "norm": {"mean": mean, "std": std},
    }, args.out)
    print(f"Saved {args.out}")

    if wb is not None:
        wb.save(args.out)
        wb.finish()


if __name__ == "__main__":
    main()
