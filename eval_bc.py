import argparse
import math
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from env import PickAndPlaceEnv
from train_bc import BCMLP, BCWithDecoder


# Feedback linearization: we control a virtual point a distance B_OFFSET ahead of the
# wheel center along the heading. Its jacobian w.r.t. (v, ω) is invertible for B>0, so
# any virtual planar acceleration (u_x, u_y) maps to a unique (v_des, ω_des). The wheel
# center then trails the offset point by B along the heading — keep B small relative to
# tolerances (PICK_RADIUS=0.06, DROP_RADIUS=0.025) but large enough that ω = lateral/B
# doesn't saturate the yaw actuator on path noise.
B_OFFSET = 0.04

# P-gain on offset-point position error. With DT=0.05 the closed-loop point dynamics
# are ẋ_b = u_x = ẋ_ref + K_P·(x_ref − x_b), so K_P sets the convergence rate. Too high
# → v_des spikes saturate K_V·(v_des−v) thrust; too low → lag behind the BC path.
K_P = 25.0

# Thrust loop: thrust = K_V·(v_des − v). The FL law produces v_des directly, not thrust,
# so this is the inner first-order velocity tracker on top.
K_V = 2.0


def track(env: PickAndPlaceEnv, path: np.ndarray, chunk_step: int) -> np.ndarray:
    """Feedback linearization of the unicycle on a time-indexed BC path.

    Define x_b = x + B·cosθ, y_b = y + B·sinθ. Then
        [ẋ_b; ẏ_b] = T(θ)·[v; ω],  T(θ) = [[cosθ, −B·sinθ], [sinθ, B·cosθ]]
    with det T = B, so the inverse exists for any θ:
        v = cosθ·u_x + sinθ·u_y,   ω = (−sinθ·u_x + cosθ·u_y) / B.
    Textbook FL drives x_b to the reference, which leaves the wheel center trailing by
    B along the heading — a 4 cm steady-state offset misses DROP_RADIUS=2.5 cm. We use
    the *wheel* error in u instead: substituting (x_ref − x) for (x_ref − x_b) makes
    v_des = v_ref·ĥ + K_P·err·ĥ (heading-projected PD on position) and ω = lateral/B.
    The wheel converges to path[k]; B only sets the relative weight of yaw to thrust.
    Path is indexed as a trajectory in time (path[k] = intended position k steps into
    the chunk); neighboring spacing is ẋ_ref — natural slow-down where expert dwelled."""
    agent = env.agent_pos.detach().cpu().numpy()
    cos_y, sin_y = math.cos(env.agent_yaw), math.sin(env.agent_yaw)
    speed = float(env.agent_speed)

    k = min(chunk_step, len(path) - 1)
    x_ref, y_ref = float(path[k, 0]), float(path[k, 1])
    if k + 1 < len(path):
        dx_ref = (path[k + 1, 0] - path[k, 0]) / env.DT
        dy_ref = (path[k + 1, 1] - path[k, 1]) / env.DT
    else:
        dx_ref = dy_ref = 0.0

    u_x = dx_ref + K_P * (x_ref - agent[0])
    u_y = dy_ref + K_P * (y_ref - agent[1])

    v_des = cos_y * u_x + sin_y * u_y
    omega_des = (-sin_y * u_x + cos_y * u_y) / B_OFFSET

    thrust_action = float(np.clip(K_V * (v_des - speed), -1.0, 1.0))
    yaw_action = float(np.clip(omega_des / env.MAX_YAW_RATE, -1.0, 1.0))
    return np.array([thrust_action, yaw_action, 0.0], dtype=np.float32)


def _build_decoder(vae_cfg: dict) -> nn.Module:
    return nn.Sequential(
        nn.Linear(vae_cfg["latent_dim"], vae_cfg["hidden"]), nn.ReLU(),
        nn.Linear(vae_cfg["hidden"], vae_cfg["hidden"]), nn.ReLU(),
        nn.Linear(vae_cfg["hidden"], vae_cfg["flat_dim"]),
    )


def load_bc(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mode = cfg.get("mode", "delta")

    if mode == "latent":
        vae_cfg = cfg["vae_cfg"]
        mlp = BCMLP(cfg["input_dim"], vae_cfg["latent_dim"], cfg["hidden"])
        decoder = _build_decoder(vae_cfg)
        model = BCWithDecoder(mlp, decoder).to(device)
    else:
        model = BCMLP(cfg["input_dim"], cfg["output_dim"], cfg["hidden"]).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device) for k, v in ckpt["norm"].items()}
    return model, cfg, norm


def predict_deltas(model: nn.Module, history: deque, cfg: dict, norm: dict, device: str) -> np.ndarray:
    """history: deque of (x, y, yaw) obs, length == cfg['history']. Returns (L, pos_dim)
    un-normalized deltas — for the (x, y, yaw) format, columns are (dx, dy, dyaw)."""
    L, P = cfg["window"], cfg["pos_dim"]
    flat = np.concatenate(list(history), axis=0).astype(np.float32)
    x = torch.from_numpy(flat).to(device).unsqueeze(0)
    x = (x - norm["obs_mean"]) / norm["obs_std"]
    with torch.no_grad():
        y = model(x).view(L, P)
    y = y * norm["delta_std"].view(1, P) + norm["delta_mean"].view(1, P)
    return y.cpu().numpy()


def run_episode(env: PickAndPlaceEnv, model: nn.Module, cfg: dict, norm: dict, device: str,
                execute: int, seed: int):
    obs, info = env.reset(seed=seed)
    H, L = cfg["history"], cfg["window"]

    history = deque([obs.copy() for _ in range(H)], maxlen=H)
    anchor = env.agent_pos.detach().cpu().numpy().copy()
    deltas = predict_deltas(model, history, cfg, norm, device)
    chunk_step = 0

    object_start = env.object_pos.detach().cpu().numpy().copy()
    poses = [(float(env.agent_pos[0]), float(env.agent_pos[1]), float(env.agent_yaw))]

    terminated = truncated = False
    while not (terminated or truncated):
        if chunk_step >= execute:
            anchor = env.agent_pos.detach().cpu().numpy().copy()
            deltas = predict_deltas(model, history, cfg, norm, device)
            chunk_step = 0

        path = anchor + deltas[:, :2]  # (L, 2) world-frame trajectory, time-indexed
        env.planned_path = path
        a = track(env, path, chunk_step)
        obs, _r, terminated, truncated, info = env.step(a)
        history.append(obs.copy())
        chunk_step += 1
        poses.append((float(env.agent_pos[0]), float(env.agent_pos[1]), float(env.agent_yaw)))

    return (bool(info["delivered"]), int(info["steps"]),
            np.array(poses, dtype=np.float32), object_start)


def _collect_overlay(env, poses, object_start, delivered):
    """Build one (image, success) entry for the grid: a viridis time-colored pose overlay."""
    cmap = plt.get_cmap("viridis")
    t = np.arange(len(poses), dtype=np.float32) / float(env.MAX_EPISODE_STEPS)
    colors = (cmap(np.clip(t, 0.0, 1.0))[:, :3] * 255.0).astype(np.float32)
    return (
        env.render_overlay(poses, object_pos=torch.from_numpy(object_start), colors=colors).cpu().numpy(),
        bool(delivered),
    )


def _save_grid(overlays, env, out_path):
    """Tile (image, success) overlays into a grid with success-bordered cells + time colorbar."""
    n = len(overlays)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    H, W, _ = overlays[0][0].shape
    pad = 2
    grid_w = cols * W + (cols + 1) * pad
    grid_h = rows * H + (rows + 1) * pad
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for i, (img, success) in enumerate(overlays):
        r, c = divmod(i, cols)
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        grid[y0:y0 + H, x0:x0 + W] = img
        border = (30, 200, 30) if success else (220, 30, 30)
        grid[y0 - pad:y0, x0 - pad:x0 + W + pad] = border
        grid[y0 + H:y0 + H + pad, x0 - pad:x0 + W + pad] = border
        grid[y0 - pad:y0 + H + pad, x0 - pad:x0] = border
        grid[y0 - pad:y0 + H + pad, x0 + W:x0 + W + pad] = border

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dpi = 100.0
    cb_band = 1.4
    fig_h = grid_h / dpi + cb_band
    fig = plt.figure(figsize=(grid_w / dpi, fig_h), dpi=dpi)
    img_frac = (grid_h / dpi) / fig_h
    ax_img = fig.add_axes([0.0, cb_band / fig_h, 1.0, img_frac])
    ax_img.imshow(grid)
    ax_img.set_axis_off()
    ax_cb = fig.add_axes([0.05, 0.55 / fig_h, 0.9, 0.25 / fig_h])
    t_max = env.MAX_EPISODE_STEPS * env.DT
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, t_max), cmap="viridis")
    cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("time (s)")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def env_eval(model: nn.Module, cfg: dict, norm: dict, device: str,
             n_episodes: int, execute: int = 8, seed_offset: int = 100000,
             viz_path: str = None) -> float:
    """Roll out n_episodes from disjoint seeds, return success rate. If viz_path is set,
    also save a pose-overlay grid at that path. Used both by the eval CLI below and by
    train_bc's in-loop evaluation."""
    env = PickAndPlaceEnv()
    env.automatic_gripper = True
    was_training = model.training
    model.eval()
    successes = 0
    overlays = [] if viz_path else None
    for ep_i in range(n_episodes):
        delivered, _steps, poses, obj0 = run_episode(
            env, model, cfg, norm, device,
            execute=execute, seed=seed_offset + ep_i,
        )
        successes += int(delivered)
        if overlays is not None:
            overlays.append(_collect_overlay(env, poses, obj0, delivered))
    if overlays:
        _save_grid(overlays, env, viz_path)
    if was_training:
        model.train()
    return successes / max(n_episodes, 1)


def main():
    p = argparse.ArgumentParser(description="Roll out a behavior-cloning policy with action chunking + a unicycle tracker, and report success rate.")
    p.add_argument("--ckpt", type=str, default="bc.pt")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=10000,
                   help="Starting eval seed. Must stay disjoint from collect_data's seeds "
                        "([--seed, --seed+episodes-1] there; defaults to [0, 99]) and from "
                        "env_eval's in-loop seeds ([100000, ...]). Default 10000 is safe "
                        "unless you collect >10000 episodes.")
    p.add_argument("--execute", type=int, default=8, help="Steps to execute before re-planning.")
    p.add_argument("--video", action="store_true",
                   help="Save videos of the first --num_video episodes into --output_dir.")
    p.add_argument("--num_video", type=int, default=1, help="Number of videos to save when --video is set.")
    p.add_argument("--viz", action="store_true",
                   help="Save a grid image (grid.png) of every rollout's poses into --output_dir.")
    p.add_argument("--output_dir", type=str, default="output_eval",
                   help="Directory for --video / --viz outputs.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, cfg, norm = load_bc(args.ckpt, args.device)
    mode = cfg.get("mode", "delta")
    print(f"Loaded BC ({mode}): history={cfg['history']} window={cfg['window']} hidden={cfg['hidden']}")
    print(f"Eval seeds: [{args.seed}, {args.seed + args.episodes - 1}] "
          f"(collect_data default range is [0, 99]; train-loop eval uses [100000, ...]).")

    env = PickAndPlaceEnv()
    env.automatic_gripper = True

    successes = 0
    step_counts = []
    overlays = []
    for i in range(args.episodes):
        if args.video and i < args.num_video:
            env._record = True
        delivered, steps, poses, object_start = run_episode(
            env, model, cfg, norm, args.device,
            execute=args.execute, seed=args.seed + i,
        )
        successes += int(delivered)
        step_counts.append(steps)
        if args.video and i < args.num_video:
            os.makedirs(args.output_dir, exist_ok=True)
            vid_path = os.path.join(args.output_dir, f"vid_{i}.mp4")
            env.save_video(vid_path)
            env._record = False
            print(f"Saved video to {vid_path}")
        if args.viz:
            cmap = plt.get_cmap("viridis")
            t = np.arange(len(poses), dtype=np.float32) / float(env.MAX_EPISODE_STEPS)
            colors = (cmap(np.clip(t, 0.0, 1.0))[:, :3] * 255.0).astype(np.float32)
            overlays.append((
                env.render_overlay(
                    poses, object_pos=torch.from_numpy(object_start), colors=colors,
                ).cpu().numpy(),
                bool(delivered),
            ))

    rate = successes / max(args.episodes, 1)
    print(f"Success: {successes}/{args.episodes} ({rate*100:.1f}%)  "
          f"avg_steps={np.mean(step_counts):.1f}  median_steps={np.median(step_counts):.0f}")

    if args.viz:
        n = len(overlays)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        H, W, _ = overlays[0][0].shape
        pad = 2
        grid_w = cols * W + (cols + 1) * pad
        grid_h = rows * H + (rows + 1) * pad
        grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
        for i, (img, success) in enumerate(overlays):
            r, c = divmod(i, cols)
            y0 = pad + r * (H + pad)
            x0 = pad + c * (W + pad)
            grid[y0:y0 + H, x0:x0 + W] = img
            border = (30, 200, 30) if success else (220, 30, 30)
            grid[y0 - pad:y0, x0 - pad:x0 + W + pad] = border
            grid[y0 + H:y0 + H + pad, x0 - pad:x0 + W + pad] = border
            grid[y0 - pad:y0 + H + pad, x0 - pad:x0] = border
            grid[y0 - pad:y0 + H + pad, x0 + W:x0 + W + pad] = border
        os.makedirs(args.output_dir, exist_ok=True)
        # Compose grid + labeled horizontal colorbar (0 ... MAX_EPISODE_STEPS * DT seconds).
        dpi = 100.0
        cb_band = 1.4  # inches reserved at the bottom for the colorbar + labels.
        fig_h = grid_h / dpi + cb_band
        fig = plt.figure(figsize=(grid_w / dpi, fig_h), dpi=dpi)
        img_frac = (grid_h / dpi) / fig_h
        ax_img = fig.add_axes([0.0, cb_band / fig_h, 1.0, img_frac])
        ax_img.imshow(grid)
        ax_img.set_axis_off()
        ax_cb = fig.add_axes([0.05, 0.55 / fig_h, 0.9, 0.25 / fig_h])
        t_max = env.MAX_EPISODE_STEPS * env.DT
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, t_max), cmap="viridis")
        cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
        cb.set_label("time (s)")
        out_path = os.path.join(args.output_dir, "grid.png")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        print(f"Saved grid to {out_path}")


if __name__ == "__main__":
    main()
