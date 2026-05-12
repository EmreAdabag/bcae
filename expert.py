import argparse
import math
import os

import numpy as np

from env import PlanarArmEnv


class ExpertController:
    """Operational-space PD controller for the planar arm.

    Computes the Cartesian force F_des = K_p · (target - ee) − K_d · ee_vel that
    would drive the end-effector toward the current task target (object before
    pickup, goal after). Maps F_des to joint torques with the transpose Jacobian
    τ = Jᵀ · F_des, and adds a joint-velocity damping term so individual joints
    settle. Null-space behavior is left to the natural damping; for redundant
    arms (N>2) this just lets the wrist drift, which is fine for the task.
    """

    def __init__(self, k_p: float = 30.0, k_d: float = 6.0, k_q: float = 1.5):
        self.k_p = float(k_p)
        self.k_d = float(k_d)
        self.k_q = float(k_q)

    def act(self, env: PlanarArmEnv) -> np.ndarray:
        if env.delivered:
            return np.zeros(env.N, dtype=np.float32)

        # Pick target: object before pickup, goal once attached.
        target_t = env.goal_pos if env.attached else env.object_pos
        target = target_t.detach().cpu().numpy().astype(np.float32) if hasattr(target_t, "detach") else np.asarray(target_t, dtype=np.float32)

        ee = np.asarray(env.ee, dtype=np.float32)
        J = env.jacobian(env.q)              # (2, N)
        ee_vel = J @ env.qd                   # Cartesian EE velocity (2,)

        # Operational-space PD: virtual force on the EE.
        F_des = self.k_p * (target - ee) - self.k_d * ee_vel  # (2,)

        # Transpose-Jacobian maps Cartesian force to joint torques.
        tau = J.T @ F_des                     # (N,)

        # Joint-space damping for stability (especially in the null-space of J).
        tau = tau - self.k_q * env.qd

        # Scale to env's [-1, 1] action range; the env multiplies by MAX_TORQUE.
        return np.clip(tau / env.MAX_TORQUE, -1.0, 1.0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Roll out the expert on a few seeds and save a video.")
    p.add_argument("--out", type=str, default="expert_rollout.mp4")
    p.add_argument("--n_joints", type=int, default=3)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = PlanarArmEnv(n_joints=args.n_joints)
    env._record = True
    ctrl = ExpertController()

    env.reset(seed=int(args.seed))

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            a = ctrl.act(env)
            _, _, terminated, truncated, info = env.step(a)
        print(f"ep {ep}: delivered={info['delivered']} steps={info['steps']}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    env.save_video(args.out)
    print(f"Saved video to {args.out}")


if __name__ == "__main__":
    main()
