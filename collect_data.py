import argparse
import os

import numpy as np

from env import PlanarArmEnv
from expert import ExpertController


def collect(num_episodes: int, seed: int, n_joints: int):
    env = PlanarArmEnv(n_joints=n_joints)
    ctrl = ExpertController()

    obs_buf, act_buf, ep_buf = [], [], []
    successes = 0

    for ep in range(num_episodes):
        if ep % 100 == 0:
            print(f"collected [{ep}] episodes")
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            a = ctrl.act(env)
            obs_buf.append(obs)
            act_buf.append(a)
            ep_buf.append(ep)
            obs, _r, terminated, truncated, info = env.step(a)
        successes += int(info["delivered"])

    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.float32),
        np.asarray(ep_buf, dtype=np.int32),
        successes,
    )


def main():
    p = argparse.ArgumentParser(description="Collect (state, action) pairs from the expert.")
    p.add_argument("--out", type=str, default="dataset.npz")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_joints", type=int, default=3)
    args = p.parse_args()

    obs, act, ep, successes = collect(
        int(args.episodes), int(args.seed), int(args.n_joints)
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.out, obs=obs, act=act, episode=ep)

    print(
        f"Saved {len(obs)} transitions across {int(args.episodes)} episodes "
        f"({successes} delivered) to {args.out}"
    )
    print(f"  obs.shape = {obs.shape}, act.shape = {act.shape}")


if __name__ == "__main__":
    main()
