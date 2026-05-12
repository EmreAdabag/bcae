import argparse
import math
import os

import numpy as np

from env import PickAndPlaceEnv


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class ExpertController:
    """Critically-damped PD on heading distance for thrust, pure-pursuit curvature for
    yaw. Designed for nonholonomic dynamics (no reverse, ω ≤ v/R_MIN).

    Thrust:  desired_accel = ω²·Δx_along − 2ω·v, clipped to [0, 1]. With no reverse,
    we can't apply negative thrust, so we layer a "coast cap" — if our remaining
    kinematic coast distance (v/k_decel, with k_decel = −ln(LIN_DAMPING)/DT) already
    exceeds the along-heading distance to target, cut throttle.

    Yaw:  pure-pursuit curvature κ = 2·sin(α)/d induces an arc that hits the target.
    The induced yaw rate is v·κ, which the env clips to its own ω ≤ v/R_MIN cap.

    Off-heading bootstrap: when the target is behind (cos α < 0) and we're too slow
    to turn (v < v_for_full_yaw), force thrust to 1 to gain rotational authority.
    """

    def __init__(self, omega: float = 2.5, k_yaw: float = 2.0):
        self.omega = float(omega)
        self.k_yaw = float(k_yaw)
        # Stateful escape: once the target falls inside the agent's natural min-turning
        # circle (an unbreakable orbit-lock), we lock to the opposite yaw direction and
        # commit to full-throttle escape until the geometry clears. Recomputing each
        # step would oscillate as the agent's heading sweeps past the cross-product
        # boundary. State is reset between episodes by detecting env.steps == 0.
        self._escape_yaw = 0.0
        self._cooldown = 0
        self._last_env_steps = -1

    def act(self, env: PickAndPlaceEnv) -> np.ndarray:
        # Detect new-episode reset.
        if env.steps == 0 or env.steps < self._last_env_steps:
            self._escape_yaw = 0.0
            self._cooldown = 0
        self._last_env_steps = env.steps

        if env.delivered:
            return np.zeros(3, dtype=np.float32)

        agent = env.agent_pos.detach().cpu().numpy()
        target_t = env.goal_pos if env.attached else env.object_pos
        target = target_t.detach().cpu().numpy()
        cos_yaw = math.cos(env.agent_yaw)
        sin_yaw = math.sin(env.agent_yaw)

        delta_real = target - agent
        dist_real = float(np.linalg.norm(delta_real))
        dot_real = cos_yaw * delta_real[0] + sin_yaw * delta_real[1]
        cross_real = cos_yaw * delta_real[1] - sin_yaw * delta_real[0]
        cos_alpha_real = dot_real / dist_real if dist_real > 1e-6 else 1.0
        sign = 1.0 if cross_real >= 0.0 else -1.0
        R = env.R_MIN
        cx = agent[0] - sign * sin_yaw * R
        cy = agent[1] + sign * cos_yaw * R
        in_natural_circle = (target[0] - cx) ** 2 + (target[1] - cy) ** 2 < R * R

        # Enter escape mode when orbit-lock is detected AND we're not already roughly
        # aimed at the target (if we are, forward motion will move us out of the lock
        # geometry without needing a U-turn). Lock the yaw direction so we commit to
        # one U-turn instead of oscillating. The cool-down counter prevents immediate
        # re-entry after an escape, letting the natural controller commit forward
        # for a while.
        if self._cooldown > 0:
            self._cooldown -= 1
        elif in_natural_circle and cos_alpha_real < 0.5 and self._escape_yaw == 0.0:
            self._escape_yaw = -sign
        # Exit when geometry clears AND we're far enough from the real target.
        if self._escape_yaw != 0.0 and not in_natural_circle and dist_real > 2.0 * R:
            self._escape_yaw = 0.0
            self._cooldown = 30

        # Wall-trap escape: if we're wedged against a wall (close to a boundary AND
        # heading into it AND moving slowly), break out by aiming for the arena
        # center — overrides everything else including target tracking. Wall
        # collision pumps the brakes via the env's 0.5× speed damping; combined
        # with min-turn-radius (ω ≤ v/R_MIN) this otherwise creates an inescapable
        # loop where the agent can't accelerate enough to rotate away.
        margin = 0.03
        near_left = agent[0] < margin and cos_yaw < -0.3
        near_right = agent[0] > 1 - margin and cos_yaw > 0.3
        near_bottom = agent[1] < margin and sin_yaw < -0.3
        near_top = agent[1] > 1 - margin and sin_yaw > 0.3
        if (near_left or near_right or near_bottom or near_top) and env.agent_speed < 0.1:
            to_center = np.array([0.5 - agent[0], 0.5 - agent[1]], dtype=np.float32)
            desired_yaw_c = math.atan2(float(to_center[1]), float(to_center[0]))
            yaw_err_c = _wrap_angle(desired_yaw_c - env.agent_yaw)
            grip = 1.0 if env.attached else (1.0 if dist_real <= env.pick_radius else 0.0)
            return np.array([1.0, float(np.sign(yaw_err_c)), grip], dtype=np.float32)

        if self._escape_yaw != 0.0:
            # Forward thrust + locked opposite yaw. Min-turn cap (env-side) handles
            # the speed-dependent ω limit; we just demand the maximum.
            grip = 1.0 if env.attached else (1.0 if dist_real <= env.pick_radius else 0.0)
            return np.array([1.0, float(self._escape_yaw), grip], dtype=np.float32)

        delta = target - agent
        dist = dist_real
        if dist < 1e-6:
            return np.array([0.0, 0.0, 1.0 if env.attached else 0.0], dtype=np.float32)

        delta_along = float(cos_yaw * delta[0] + sin_yaw * delta[1])
        cos_alpha = delta_along / dist

        # Yaw: proportional on heading-to-target angle. The env's own ω ≤ v/R_MIN cap
        # clamps the rate when v is low — we just hand it the unconstrained desire.
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        err = _wrap_angle(desired_yaw - env.agent_yaw)
        yaw_action = float(np.clip(self.k_yaw * err, -1.0, 1.0))

        # Thrust: critically-damped PD on along-heading distance. Negative thrust is an
        # active brake; the env's kinematic clamp pins agent_speed ≥ 0 so we can't reverse.
        desired_accel = self.omega * self.omega * delta_along - 2.0 * self.omega * float(env.agent_speed)
        thrust_action = float(np.clip(env.MASS * desired_accel / env.MAX_THRUST, -1.0, 1.0))

        # Off-heading bootstrap: at v=0 we can't turn (ω ≤ v/R_MIN), so commit forward
        # thrust to gain rotational authority. Use brake-to-stop once pointed back.
        if cos_alpha < 0:
            v_for_full_yaw = env.MAX_YAW_RATE * env.R_MIN
            if float(env.agent_speed) < v_for_full_yaw:
                thrust_action = 1.0

        # Gripper as before.
        if env.attached:
            grip_action = 1.0
        elif env.gripper_closed:
            grip_action = 0.0
        else:
            d_obj = float(np.linalg.norm(env.object_pos.detach().cpu().numpy() - agent))
            grip_action = 1.0 if d_obj <= env.pick_radius else 0.0

        return np.array([thrust_action, yaw_action, grip_action], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Roll out expert policy and save a video")
    parser.add_argument("--out", type=str, default="expert_rollout.mp4")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = PickAndPlaceEnv()
    env._record = True
    ctrl = ExpertController()

    _obs, info = env.reset(seed=int(args.seed))

    for ep in range(2):
        env.reset()

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            a = ctrl.act(env)
            _obs, _r, terminated, truncated, info = env.step(a)
            steps += 1

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    env.save_video(args.out)

    print(f"Saved video to {args.out}")


if __name__ == "__main__":
    main()
