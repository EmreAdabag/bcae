import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_pairs(obs: np.ndarray, ep: np.ndarray, history: int, window: int):
    """For each anchor t with H obs of history and L future (x, y, yaw) positions inside
    the same episode, build (obs_history, delta_traj) pairs.

    obs_history: obs[t-H+1 : t+1].flatten()  -> (H * obs_dim,)   full env obs
    delta_traj:  pos[t : t+L] - pos[t]       -> (L, 3)           unwrapped yaw -> continuous deltas
    """
    obs = obs.astype(np.float32)
    # (x, y, yaw) from env's _get_obs layout (x, y, cos_yaw, sin_yaw, ...).
    pos = np.stack([obs[:, 0], obs[:, 1], np.arctan2(obs[:, 3], obs[:, 2])], axis=1).astype(np.float32)
    H, L = history, window
    starts = np.where(np.concatenate([[True], ep[1:] != ep[:-1]]))[0]
    ends = np.concatenate([starts[1:], [len(ep)]])
    inputs, targets = [], []
    for s, e in zip(starts, ends):
        pos[s:e, 2] = np.unwrap(pos[s:e, 2])
        if e - s < H + L - 1:
            continue
        for t in range(s + H - 1, e - L + 1):
            inputs.append(obs[t - H + 1 : t + 1].reshape(-1))
            targets.append(pos[t : t + L] - pos[t])
    return np.stack(inputs, 0), np.stack(targets, 0)


class BCMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class BCWithDecoder(nn.Module):
    """Composition of a trainable MLP head (predicting a latent) with a frozen VAE decoder
    that maps the latent back to a flat z-scored delta-trajectory."""

    def __init__(self, mlp: nn.Module, decoder: nn.Module):
        super().__init__()
        self.mlp = mlp
        self.decoder = decoder

    def forward(self, x):
        z = self.mlp(x)
        return self.decoder(z)


def build_vae_decoder(vae_cfg: dict, vae_state: dict, device: str) -> nn.Module:
    """Reconstruct just the VAE.dec submodule from the VAE's flat state_dict and freeze it."""
    dec = nn.Sequential(
        nn.Linear(vae_cfg["latent_dim"], vae_cfg["hidden"]), nn.ReLU(),
        nn.Linear(vae_cfg["hidden"], vae_cfg["hidden"]), nn.ReLU(),
        nn.Linear(vae_cfg["hidden"], vae_cfg["flat_dim"]),
    )
    dec_state = {k.split(".", 1)[1]: v for k, v in vae_state.items() if k.startswith("dec.")}
    dec.load_state_dict(dec_state)
    for p in dec.parameters():
        p.requires_grad = False
    return dec.to(device).eval()


def build_vae_encoder(vae_cfg: dict, vae_state: dict, device: str) -> nn.Module:
    """Reconstruct VAE.enc + fc_mu (deterministic latent-mean encoder) and freeze it. Used
    to supervise the BC MLP in latent space instead of backpropagating through the decoder:
    one trajectory maps to one well-defined μ, while many z's would decode near the target."""
    enc = nn.Sequential(
        nn.Linear(vae_cfg["flat_dim"], vae_cfg["hidden"]), nn.ReLU(),
        nn.Linear(vae_cfg["hidden"], vae_cfg["hidden"]), nn.ReLU(),
        nn.Linear(vae_cfg["hidden"], vae_cfg["latent_dim"]),
    )
    state = {}
    for k, v in vae_state.items():
        if k.startswith("enc."):
            state[k.split(".", 1)[1]] = v
        elif k.startswith("fc_mu."):
            state[f"4.{k.split('.', 1)[1]}"] = v
    enc.load_state_dict(state)
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device).eval()


def main():
    p = argparse.ArgumentParser(description="Train an MLP behavior-cloning policy that maps a short obs history to a future delta-position trajectory. Optionally regress through a frozen pretrained VAE decoder.")
    p.add_argument("--data", type=str, default="dataset.npz")
    p.add_argument("--out", type=str, default="bc.pt")
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=10000,
                   help="Total number of gradient steps. Data loader is restarted as needed.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-off", dest="wandb", action="store_false", help="Pass to disable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="onceler-bc",
                   help="wandb project name (use a unique suffix to keep parallel sweeps separate).")
    p.add_argument("--vae_ckpt", type=str, default=None,
                   help="If set, the MLP outputs the VAE's latent and is composed with a frozen pretrained decoder.")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Limit training to the first N episode IDs from the dataset.")
    p.add_argument("--eval_episodes", type=int, default=30,
                   help="Env-rollout episodes per eval point (0 disables in-loop evaluation).")
    p.add_argument("--eval_every", type=int, default=5000,
                   help="Run env evaluation every N gradient steps (and at the final step). Default gives ~2 evals per run at max_steps=10000 (mid + end).")
    p.add_argument("--viz_dir", type=str, default=None,
                   help="If set, write a pose-overlay grid per eval point into this directory.")
    p.add_argument("--tag", type=str, default=None, help="Optional tag for the wandb run name.")
    p.add_argument("--log_csv", type=str, default=None,
                   help="Where to write the per-eval-point CSV log. Defaults to <out>.csv.")
    args = p.parse_args()

    wb = None
    if args.wandb:
        import wandb as wb
        wb.init(project=args.wandb_project, name=args.tag, config=vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data)
    obs, ep = data["obs"], data["episode"]
    if args.max_episodes is not None:
        keep = ep < args.max_episodes
        obs, ep = obs[keep], ep[keep]
    obs_dim = obs.shape[1]
    print(f"Using {len(obs)} transitions from {len(np.unique(ep))} episodes (obs_dim={obs_dim})")

    # make_pairs always returns (dx, dy, dyaw) targets; we slice to pos_dim per mode below.
    inputs, targets = make_pairs(obs, ep, args.history, args.window)
    print(f"Built {len(inputs)} pairs: in={inputs.shape[1:]} full out={targets.shape[1:]}")

    perm = np.random.permutation(len(inputs))
    n_val = int(len(inputs) * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    tr_in, va_in = inputs[train_idx], inputs[val_idx]
    tr_tg_full, va_tg_full = targets[train_idx], targets[val_idx]

    obs_mean = tr_in.mean(axis=0)
    obs_std = tr_in.std(axis=0) + 1e-6

    # Latent-mode setup: load VAE, build frozen encoder+decoder, override target normalization.
    # The encoder supplies deterministic z targets for supervision; the decoder is only used
    # at rollout time (env_eval and eval_bc.py) to turn predicted z back into a trajectory.
    decoder = None
    encoder = None
    mode = "delta"
    vae_cfg_to_save = None
    if args.vae_ckpt:
        vae_ckpt = torch.load(args.vae_ckpt, map_location=args.device, weights_only=False)
        vae_cfg = vae_ckpt["config"]
        assert vae_cfg["window"] == args.window, "VAE window must match BC window."
        pos_dim = vae_cfg.get("pos_dim") or vae_cfg["flat_dim"] // vae_cfg["window"]
        decoder = build_vae_decoder(vae_cfg, vae_ckpt["state_dict"], args.device)
        encoder = build_vae_encoder(vae_cfg, vae_ckpt["state_dict"], args.device)
        delta_mean = np.asarray(vae_ckpt["norm"]["mean"], dtype=np.float32)
        delta_std = np.asarray(vae_ckpt["norm"]["std"], dtype=np.float32)
        output_dim = vae_cfg["latent_dim"]
        mode = "latent"
        vae_cfg_to_save = vae_cfg
        print(f"Latent mode: pos_dim={pos_dim}, z={vae_cfg['latent_dim']} (from {args.vae_ckpt})")
    else:
        # Base mode predicts (dx, dy) only. Yaw deltas have ~6x the std of xy deltas
        # (turn-in-place spikes hit ~3 rad), so on small datasets they dominate the
        # z-scored loss and crowd out xy capacity. Yaw isn't used at rollout anyway.
        pos_dim = 2
        delta_mean = tr_tg_full.reshape(-1, 3)[:, :pos_dim].mean(axis=0)
        delta_std = tr_tg_full.reshape(-1, 3)[:, :pos_dim].std(axis=0) + 1e-6
        output_dim = args.window * pos_dim

    tr_tg = tr_tg_full[..., :pos_dim]
    va_tg = va_tg_full[..., :pos_dim]

    target_dim = args.window * pos_dim
    tr_in_n = (tr_in - obs_mean) / obs_std
    va_in_n = (va_in - obs_mean) / obs_std
    tr_tg_n = (tr_tg - delta_mean) / delta_std
    va_tg_n = (va_tg - delta_mean) / delta_std

    tr_tg_flat = tr_tg_n.reshape(-1, target_dim).astype(np.float32)
    va_tg_flat = va_tg_n.reshape(-1, target_dim).astype(np.float32)

    if mode == "latent":
        # Encode the z-scored ground-truth trajectory to deterministic μ — this is the
        # supervision target. No grad flows through the encoder or decoder at train time.
        def _encode_all(flat: np.ndarray) -> np.ndarray:
            out = []
            with torch.no_grad():
                for i in range(0, len(flat), 4096):
                    chunk = torch.from_numpy(flat[i:i + 4096]).to(args.device)
                    out.append(encoder(chunk).cpu().numpy())
            return np.concatenate(out, axis=0) if out else np.zeros((0, vae_cfg["latent_dim"]), dtype=np.float32)
        tr_y_np = _encode_all(tr_tg_flat)
        va_y_np = _encode_all(va_tg_flat)
        print(f"Encoded {len(tr_y_np)} train / {len(va_y_np)} val trajectories -> z (dim={tr_y_np.shape[1]})")
    else:
        tr_y_np = tr_tg_flat
        va_y_np = va_tg_flat

    train_x = torch.from_numpy(tr_in_n).float()
    train_y = torch.from_numpy(tr_y_np).float()
    val_x = torch.from_numpy(va_in_n).float()
    val_y = torch.from_numpy(va_y_np).float()

    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y),
                            batch_size=args.batch_size, shuffle=False)

    input_dim = args.history * obs_dim
    mlp = BCMLP(input_dim, output_dim, args.hidden).to(args.device)
    if mode == "latent":
        model = BCWithDecoder(mlp, decoder).to(args.device)
    else:
        model = mlp
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    norm_t = {
        "obs_mean": torch.from_numpy(obs_mean.astype(np.float32)).to(args.device),
        "obs_std": torch.from_numpy(obs_std.astype(np.float32)).to(args.device),
        "delta_mean": torch.from_numpy(np.asarray(delta_mean, dtype=np.float32)).to(args.device),
        "delta_std": torch.from_numpy(np.asarray(delta_std, dtype=np.float32)).to(args.device),
    }
    cfg_for_eval = {"history": args.history, "window": args.window, "pos_dim": pos_dim}

    def run_validation():
        model.eval()
        with torch.no_grad():
            tot = 0.0
            for xb, yb in val_loader:
                xb = xb.to(args.device); yb = yb.to(args.device)
                # In latent mode mlp(xb) outputs z; yb is the encoded target z. In delta
                # mode this is just the flat-trajectory MSE.
                pred = mlp(xb)
                tot += F.mse_loss(pred, yb, reduction="none").sum(dim=1).mean().item()
            return tot / max(len(val_loader), 1)

    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    log_rows = []
    global_step = 0
    run_loss = 0.0
    run_count = 0
    steps_per_epoch = max(len(train_loader), 1)
    model.train()
    done = False
    while not done:
        for xb, yb in train_loader:
            if global_step >= args.max_steps:
                done = True
                break
            xb = xb.to(args.device); yb = yb.to(args.device)
            pred = mlp(xb)
            loss = F.mse_loss(pred, yb, reduction="none").sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item()
            run_count += 1
            global_step += 1
            epoch = (global_step - 1) // steps_per_epoch + 1

            if global_step % args.log_every == 0 or global_step == args.max_steps:
                tr_loss = run_loss / run_count
                run_loss = 0.0; run_count = 0
                vl_loss = run_validation()
                model.train()
                print(f"step {global_step:6d} (epoch {epoch:3d})  train mse={tr_loss:.4f}   val mse={vl_loss:.4f}")
                if wb is not None:
                    wb.log({"epoch": epoch, "train/mse": tr_loss, "val/mse": vl_loss}, step=global_step)

            if args.eval_episodes > 0 and (global_step % args.eval_every == 0 or global_step == args.max_steps):
                from eval_bc import env_eval  # lazy to break the train_bc <-> eval_bc cycle
                viz_path = (os.path.join(args.viz_dir, f"step_{global_step:06d}.png")
                            if args.viz_dir else None)
                success = env_eval(model, cfg_for_eval, norm_t, args.device,
                                   args.eval_episodes, viz_path=viz_path)
                model.train()
                print(f"   [step {global_step:6d}] env eval: {success*100:.1f}% over {args.eval_episodes} episodes"
                      + (f"  viz={viz_path}" if viz_path else ""))
                log_rows.append({"epoch": epoch, "step": global_step, "success": success})
                if wb is not None:
                    wb.log({"eval/success": success, "epoch": epoch}, step=global_step)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    config = {
        "history": args.history,
        "window": args.window,
        "obs_dim": obs_dim,
        "pos_dim": pos_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden": args.hidden,
        "mode": mode,
    }
    if vae_cfg_to_save is not None:
        config["vae_cfg"] = vae_cfg_to_save
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "norm": {
            "obs_mean": obs_mean, "obs_std": obs_std,
            "delta_mean": np.asarray(delta_mean), "delta_std": np.asarray(delta_std),
        },
    }, args.out)
    print(f"Saved {args.out}")

    log_csv_path = args.log_csv or os.path.splitext(args.out)[0] + ".csv"
    if log_rows:
        with open(log_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "step", "success"])
            w.writeheader()
            for r in log_rows:
                w.writerow(r)
        print(f"Saved {log_csv_path}")

    if wb is not None:
        wb.save(args.out)
        wb.finish()


if __name__ == "__main__":
    main()
