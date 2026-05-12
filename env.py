import os
import json
import shutil
import subprocess
from collections import deque
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class PlanarArmEnv(gym.Env):
    """Planar N-link revolute arm, no gravity, pick-and-place.

    Configurable via constructor arg `n_joints` (default 3). All links share
    length `1.0 / n_joints` so total reach is 1.0; the base sits at the arena
    center so the workspace covers all of [0, 1]².

    Action (Box[N]):  per-joint torque in [-1, 1], scaled by MAX_TORQUE.
    Observation (Box[2N + 9]):
        [ee_x, ee_y, cos(theta_N), sin(theta_N),
         q_1, ..., q_N, qd_1, ..., qd_N,
         object_x, object_y, goal_x, goal_y, attached]
    (theta_i is the world-frame orientation of link i = cumsum(q)[i-1].)

    Auto-gripper: closes whenever the EE is within PICK_RADIUS of the object.
    Termination: object stays within DROP_RADIUS of the goal for
    GOAL_DWELL_STEPS consecutive steps.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    DT = 0.05
    IMAGE_SIZE = 256
    MAX_TORQUE = 1.0
    JOINT_DAMPING = 0.5
    JOINT_INERTIA = 1.0
    BASE_POS = (0.5, 0.5)

    PICK_RADIUS = 0.06
    DROP_RADIUS = 0.025
    GOAL_DWELL_STEPS = 10
    MAX_EPISODE_STEPS = 400

    OBJECT_RADIUS = 0.04
    GOAL_RADIUS = 0.06
    GOAL_OUTLINE = 0.03
    JOINT_RADIUS = 0.015
    EE_RADIUS = 0.02
    LINK_WIDTH = 0.012
    TRAIL_LEN = 5

    BG_COLOR = (245, 245, 245)
    ARM_COLOR = (220, 30, 30)
    JOINT_COLOR = (40, 40, 40)
    OBJECT_COLOR = (30, 60, 220)
    GOAL_COLOR = (30, 230, 60)

    def __init__(self, n_joints: int = 3):
        super().__init__()
        self.render_mode = "rgb_array"
        self.dtype = torch.float32
        self.N = int(n_joints)
        self.link_length = 1.0 / self.N
        self.base_pos = np.array(self.BASE_POS, dtype=np.float32)

        # Convenience mirrors used by callers/tools.
        self.dt = self.DT
        self.H = self.IMAGE_SIZE
        self.W = self.IMAGE_SIZE
        self.max_episode_steps = self.MAX_EPISODE_STEPS
        self.pick_radius = self.PICK_RADIUS
        self.drop_radius = self.DROP_RADIUS
        self.goal_dwell_steps = self.GOAL_DWELL_STEPS

        self._bg = torch.tensor(self.BG_COLOR, dtype=torch.float32)
        self._arm_c = torch.tensor(self.ARM_COLOR, dtype=torch.float32)
        self._joint_c = torch.tensor(self.JOINT_COLOR, dtype=torch.float32)
        self._obj_c = torch.tensor(self.OBJECT_COLOR, dtype=torch.float32)
        self._goal_c = torch.tensor(self.GOAL_COLOR, dtype=torch.float32)

        ys = torch.linspace(0.0, 1.0, self.H)
        xs = torch.linspace(0.0, 1.0, self.W)
        self.grid_y, self.grid_x = torch.meshgrid(ys, xs, indexing="ij")

        self.action_space = spaces.Box(
            low=-1.0 * np.ones(self.N, dtype=np.float32),
            high=1.0 * np.ones(self.N, dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = 2 * self.N + 9
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high=np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # State (filled in by reset).
        self.q = None
        self.qd = None
        self.ee = None
        self.object_pos = None
        self.goal_pos = None
        self.attached = False
        self.dwell_counter = 0
        self.delivered = False
        self.steps = 0

        # Trail of past EE positions (oldest -> newest), excluding the current pose.
        self._trail: deque = deque(maxlen=self.TRAIL_LEN)

        # Recording / debug-overlay machinery, kept compatible with the old env's API.
        self._record = False
        self._frames: list[torch.Tensor] = []
        self.planned_path: Optional[np.ndarray] = None
        self.automatic_gripper = True

    # ---- kinematics ----
    def fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: joint angles -> EE position."""
        thetas = np.cumsum(q)
        x = self.base_pos[0] + self.link_length * float(np.sum(np.cos(thetas)))
        y = self.base_pos[1] + self.link_length * float(np.sum(np.sin(thetas)))
        return np.array([x, y], dtype=np.float32)

    def joint_positions(self, q: np.ndarray) -> np.ndarray:
        """Return positions of base + all joint pivots + EE, shape (N+1, 2)."""
        thetas = np.cumsum(q)
        out = np.zeros((self.N + 1, 2), dtype=np.float32)
        out[0] = self.base_pos
        for i in range(self.N):
            out[i + 1, 0] = out[i, 0] + self.link_length * np.cos(thetas[i])
            out[i + 1, 1] = out[i, 1] + self.link_length * np.sin(thetas[i])
        return out

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """∂(ee_x, ee_y) / ∂q, shape (2, N). Standard planar-arm Jacobian."""
        thetas = np.cumsum(q)
        # ee = sum_i L * (cos θ_i, sin θ_i). θ_i depends on q_1..q_i, so:
        # ∂ee_x/∂q_j = sum_{i ≥ j} -L sin θ_i,   ∂ee_y/∂q_j = sum_{i ≥ j} L cos θ_i.
        sin_t = self.link_length * np.sin(thetas)
        cos_t = self.link_length * np.cos(thetas)
        # cumulative sums from the right.
        rev_sin = np.cumsum(sin_t[::-1])[::-1]
        rev_cos = np.cumsum(cos_t[::-1])[::-1]
        return np.stack([-rev_sin, rev_cos], axis=0).astype(np.float32)

    # ---- gymnasium API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        def r2():
            return self.np_random.uniform(0.0, 1.0, size=2).astype(np.float32)

        # Random initial joint config; random object/goal in [0,1]², rejection-sampled
        # to be non-overlapping with each other and with the initial EE position.
        self.q = self.np_random.uniform(-np.pi, np.pi, size=self.N).astype(np.float32)
        self.qd = np.zeros(self.N, dtype=np.float32)
        self.ee = self.fk(self.q)

        op = r2()
        while np.linalg.norm(op - self.ee) <= self.PICK_RADIUS:
            op = r2()
        gp = r2()
        while np.linalg.norm(op - gp) <= 2.0 * (self.OBJECT_RADIUS + self.GOAL_RADIUS):
            gp = r2()

        self.object_pos = torch.from_numpy(op).float()
        self.goal_pos = torch.from_numpy(gp).float()
        self.attached = False
        self.dwell_counter = 0
        self.delivered = False
        self.steps = 0
        self._trail.clear()

        if not self._record:
            self._frames = []
        else:
            self._frames.append(self.current_frame())

        return self._get_obs(), self._get_info()

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        tau = np.clip(a[: self.N], -1.0, 1.0) * self.MAX_TORQUE

        # Second-order joint dynamics: qdd = (tau - b*qd) / I. Euler integration.
        qdd = (tau - self.JOINT_DAMPING * self.qd) / self.JOINT_INERTIA
        self.qd = (self.qd + qdd * self.DT).astype(np.float32)
        self.q = ((self.q + self.qd * self.DT + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)

        # Trail of past EE positions for rendering.
        self._trail.append(torch.from_numpy(self.ee.copy()))
        self.ee = self.fk(self.q)

        # Auto-gripper: attach once EE is inside PICK_RADIUS of the object.
        if not self.attached:
            d_obj = float(np.linalg.norm(self.ee - self.object_pos.numpy()))
            if d_obj <= self.PICK_RADIUS:
                self.attached = True
        if self.attached:
            self.object_pos = torch.from_numpy(self.ee.copy()).float()

        # Goal-dwell termination.
        d_goal = float(np.linalg.norm(self.object_pos.numpy() - self.goal_pos.numpy()))
        if d_goal <= self.DROP_RADIUS:
            self.dwell_counter += 1
        else:
            self.dwell_counter = 0
        if self.dwell_counter >= self.GOAL_DWELL_STEPS:
            self.delivered = True

        terminated = bool(self.delivered)
        self.steps += 1
        truncated = self.steps >= self.MAX_EPISODE_STEPS

        if self._record:
            self._frames.append(self.current_frame())

        # 0 reward — we score on info["delivered"].
        return self._get_obs(), 0.0, terminated, truncated, self._get_info()

    def _get_obs(self):
        thetas = np.cumsum(self.q)
        ee_yaw = float(thetas[-1])
        return np.concatenate([
            self.ee,
            np.array([np.cos(ee_yaw), np.sin(ee_yaw)], dtype=np.float32),
            self.q,
            self.qd,
            self.object_pos.numpy(),
            self.goal_pos.numpy(),
            np.array([1.0 if self.attached else 0.0], dtype=np.float32),
        ]).astype(np.float32)

    def _get_info(self):
        return {
            "delivered": bool(self.delivered),
            "steps": int(self.steps),
            "attached": bool(self.attached),
            "ee_pos": self.ee.tolist(),
        }

    # ---- backwards-compat properties for external callers (eval_bc.track etc.) ----
    @property
    def agent_pos(self):
        # The "agent" for the policy is the EE.
        return torch.from_numpy(self.ee.copy())

    @property
    def agent_yaw(self):
        thetas = np.cumsum(self.q)
        return float(thetas[-1])

    @property
    def agent_speed(self):
        # Cartesian EE speed magnitude.
        J = self.jacobian(self.q)
        v = J @ self.qd
        return float(np.linalg.norm(v))

    # ---- rendering ----
    def _seg_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-pixel distance from each grid point to the segment a→b."""
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        vx, vy = bx - ax, by - ay
        vv = max(vx * vx + vy * vy, 1e-9)
        px = self.grid_x - ax
        py = self.grid_y - ay
        t = ((px * vx + py * vy) / vv).clamp(0.0, 1.0)
        proj_x = ax + t * vx
        proj_y = ay + t * vy
        return torch.sqrt((self.grid_x - proj_x) ** 2 + (self.grid_y - proj_y) ** 2)

    def current_frame(self) -> torch.Tensor:
        """RGB image of the current arm + object + goal, shape (H, W, 3) uint8."""
        frame = self._bg.unsqueeze(0).unsqueeze(0).expand(self.H, self.W, 3).clone()

        # Goal annulus
        gx, gy = float(self.goal_pos[0]), float(self.goal_pos[1])
        gd2 = (self.grid_x - gx) ** 2 + (self.grid_y - gy) ** 2
        gr_outer = self.GOAL_RADIUS
        gr_inner = max(self.GOAL_RADIUS - self.GOAL_OUTLINE, 0.0)
        goal_mask = (gd2 <= gr_outer ** 2) & (gd2 >= gr_inner ** 2)
        frame[goal_mask] = self._goal_c

        # Arm links
        joints = self.joint_positions(self.q)
        for i in range(self.N):
            a = torch.tensor(joints[i], dtype=torch.float32)
            b = torch.tensor(joints[i + 1], dtype=torch.float32)
            mask = self._seg_distance(a, b) <= self.LINK_WIDTH
            frame[mask] = self._arm_c

        # Joint pivots (incl. base)
        for j in joints:
            d2 = (self.grid_x - float(j[0])) ** 2 + (self.grid_y - float(j[1])) ** 2
            frame[d2 <= self.JOINT_RADIUS ** 2] = self._joint_c

        # Object
        ox, oy = float(self.object_pos[0]), float(self.object_pos[1])
        od2 = (self.grid_x - ox) ** 2 + (self.grid_y - oy) ** 2
        frame[od2 <= self.OBJECT_RADIUS ** 2] = self._obj_c

        return (frame * 1.0).to(torch.uint8) if frame.dtype != torch.uint8 else frame

    def render_overlay(self, poses, object_pos=None, goal_pos=None, colors=None) -> torch.Tensor:
        """Render a static background (object + goal) with a colored EE-trajectory
        polyline overlaid. `poses` is a sequence of (x, y, yaw) tuples; we ignore yaw
        and just dot the EE positions.

        Object/goal default to the env's current state; pass overrides to pin them
        across episodes. `colors` is an optional (T, 3) float array in [0, 255] for
        per-pose dots (e.g. time-coloring with viridis)."""
        frame = self._bg.unsqueeze(0).unsqueeze(0).expand(self.H, self.W, 3).clone().float()

        gp = self.goal_pos if goal_pos is None else (goal_pos.detach().cpu().numpy()
                                                    if isinstance(goal_pos, torch.Tensor) else goal_pos)
        op = self.object_pos if object_pos is None else (object_pos.detach().cpu().numpy()
                                                        if isinstance(object_pos, torch.Tensor) else object_pos)
        gx, gy = float(gp[0]), float(gp[1])
        gd2 = (self.grid_x - gx) ** 2 + (self.grid_y - gy) ** 2
        gr_outer = self.GOAL_RADIUS
        gr_inner = max(self.GOAL_RADIUS - self.GOAL_OUTLINE, 0.0)
        goal_mask = (gd2 <= gr_outer ** 2) & (gd2 >= gr_inner ** 2)
        frame[goal_mask] = self._goal_c

        ox, oy = float(op[0]), float(op[1])
        od2 = (self.grid_x - ox) ** 2 + (self.grid_y - oy) ** 2
        frame[od2 <= self.OBJECT_RADIUS ** 2] = self._obj_c

        # Base marker
        bx, by = float(self.base_pos[0]), float(self.base_pos[1])
        bd2 = (self.grid_x - bx) ** 2 + (self.grid_y - by) ** 2
        frame[bd2 <= self.JOINT_RADIUS ** 2] = self._joint_c

        # EE trail
        dot_r = 0.006
        for i, p in enumerate(poses):
            px = float(p[0]); py = float(p[1])
            d2 = (self.grid_x - px) ** 2 + (self.grid_y - py) ** 2
            mask = d2 <= dot_r ** 2
            if colors is not None and i < len(colors):
                c = torch.tensor(colors[i], dtype=torch.float32)
            else:
                c = self._arm_c
            frame[mask] = c

        return frame.to(torch.uint8)

    def save_video(self, path: str, fps: Optional[int] = None) -> None:
        if not self._frames:
            return
        fps = max(1, int(round(1.0 / self.DT))) if fps is None else int(fps)
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Lazy: write frames as PNGs and use ffmpeg if available, else just save first frame.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            for i, f in enumerate(self._frames):
                arr = f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(tmp, f"f_{i:05d}.png"), arr)
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                # Just keep the last frame as a still image.
                last = self._frames[-1].cpu().numpy() if hasattr(self._frames[-1], "cpu") else np.asarray(self._frames[-1])
                import matplotlib.pyplot as plt
                plt.imsave(path.replace(".mp4", ".png"), last)
                return
            subprocess.run([
                ffmpeg, "-y", "-framerate", str(fps),
                "-i", os.path.join(tmp, "f_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
            ], check=False, capture_output=True)


# Keep the old import name available so existing code keeps working.
PickAndPlaceEnv = PlanarArmEnv
