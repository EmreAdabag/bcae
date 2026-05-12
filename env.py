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


class PickAndPlaceEnv(gym.Env):
    """
    2D point-mass pick-and-place with unicycle motion.

    Action (Box[3]):  [thrust in [-1, 1], yaw_rate in [-1, 1], gripper in [0, 1]].
        thrust   - forward force along the agent heading
        yaw_rate - turning rate
        gripper  - >0.5 means closed; closing within `PICK_RADIUS` of the object
                   attaches it. Opening releases the object in place.

    Observation (Box[9]):
        [agent_x, agent_y, cos(yaw), sin(yaw),
         object_x, object_y, goal_x, goal_y, attached]

    Termination: object stays within `DROP_RADIUS` of the goal for
    `GOAL_DWELL_STEPS` consecutive steps.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    # --- hard-coded configuration ---
    DT = 0.05
    IMAGE_SIZE = 256
    MASS = 1.0
    OBJECT_MASS = 0.0
    LIN_DAMPING = 0.99
    MAX_THRUST = 1.0
    MAX_YAW_RATE = 5.0
    # Nonholonomic constraints: no reverse (thrust ≥ 0) and a minimum turning radius
    # |ω| ≤ |v| / R_MIN. At v=0 the agent cannot turn; the path manifold becomes
    # Dubins-like, so sloppy raw-Δ predictions with sharp kinks become untrackable.
    R_MIN = 0.10
    MAX_EPISODE_STEPS = 400

    AGENT_RADIUS = 0.025
    OBJECT_RADIUS = 0.04
    GOAL_RADIUS = 0.06
    GOAL_OUTLINE = 0.03
    PICK_RADIUS = 0.06
    DROP_RADIUS = 0.025
    GOAL_DWELL_STEPS = 10
    TRAIL_LEN = 5

    BG_COLOR = (245, 245, 245)
    AGENT_COLOR = (220, 30, 30)
    YAW_COLOR = (0, 0, 0)            # black
    GRIP_DOT_COLOR = (255, 255, 255)
    OBJECT_COLOR = (30, 60, 220)
    GOAL_COLOR = (30, 230, 60)

    def __init__(self):
        super().__init__()
        self.render_mode = "rgb_array"
        self.dtype = torch.float32

        # Convenience public mirrors of constants used by callers.
        self.dt = self.DT
        self.H = self.IMAGE_SIZE
        self.W = self.IMAGE_SIZE
        self.max_episode_steps = self.MAX_EPISODE_STEPS
        self.pick_radius = self.PICK_RADIUS
        self.drop_radius = self.DROP_RADIUS
        self.goal_dwell_steps = self.GOAL_DWELL_STEPS

        self._bg = torch.tensor(self.BG_COLOR, dtype=torch.float32)
        self._agent_c = torch.tensor(self.AGENT_COLOR, dtype=torch.float32)
        self._yaw_c = torch.tensor(self.YAW_COLOR, dtype=torch.float32)
        self._grip_c = torch.tensor(self.GRIP_DOT_COLOR, dtype=torch.float32)
        self._obj_c = torch.tensor(self.OBJECT_COLOR, dtype=torch.float32)
        self._goal_c = torch.tensor(self.GOAL_COLOR, dtype=torch.float32)

        ys = torch.linspace(0.0, 1.0, self.H)
        xs = torch.linspace(0.0, 1.0, self.W)
        self.grid_y, self.grid_x = torch.meshgrid(ys, xs, indexing="ij")

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        obs_low = np.array(
            [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        obs_high = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # State (filled in by reset).
        self.agent_pos = None
        self.agent_yaw = None
        self.agent_speed = None
        self.object_pos = None
        self.goal_pos = None
        self.attached = False
        self.gripper_closed = False
        self.dwell_counter = 0
        self.delivered = False
        self.steps = 0

        # Trail of past agent positions (oldest -> newest), excluding the current pose.
        self._trail: deque = deque(maxlen=self.TRAIL_LEN)

        # Recording: flip env._record = True before reset() to capture frames.
        self._record = False
        self._frames: list[torch.Tensor] = []

        # Optional debug overlay: future-trajectory dots + arrows. Set per-step by the
        # caller, persists until reassigned/None. Accepts (N, 2) of (x, y) positions —
        # arrows are inferred from the vector to the next point — or (N, 3) of
        # (x, y, yaw) for explicit headings.
        self.planned_path: Optional[np.ndarray] = None

        # When True, the gripper action is ignored: the env closes the gripper
        # automatically whenever the agent is within PICK_RADIUS of the object
        # (and stays closed while attached).
        self.automatic_gripper = True

    # ---- gymnasium API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        def r2():
            return self.np_random.uniform(0.0, 1.0, size=2).astype(np.float32)

        ap = r2()
        op = r2()
        while np.linalg.norm(ap - op) <= self.PICK_RADIUS:
            op = r2()
        gp = r2()

        self.agent_pos = torch.tensor(ap, dtype=self.dtype)
        self.object_pos = torch.tensor(op, dtype=self.dtype)
        self.goal_pos = torch.tensor(gp, dtype=self.dtype)
        self.agent_yaw = float(self.np_random.uniform(-np.pi, np.pi))
        self.agent_speed = 0.0
        self.attached = False
        self.gripper_closed = False
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
        # Thrust is a longitudinal force: negative = active brake. The "no reverse"
        # constraint is kinematic — agent_speed gets clamped at 0 after integration,
        # so the agent can decelerate quickly but cannot end up moving backward.
        thrust = float(np.clip(a[0], -1.0, 1.0)) * self.MAX_THRUST
        yaw_rate = float(np.clip(a[1], -1.0, 1.0)) * self.MAX_YAW_RATE
        # Min-turning-radius cap: |ω| ≤ |v| / R_MIN (cannot spin in place at v=0).
        omega_cap = abs(self.agent_speed) / self.R_MIN
        yaw_rate = float(np.clip(yaw_rate, -omega_cap, omega_cap))
        if self.automatic_gripper:
            if self.attached:
                grip_cmd = True
            else:
                d_obj = float(torch.linalg.vector_norm(self.agent_pos - self.object_pos))
                grip_cmd = d_obj <= self.PICK_RADIUS
        else:
            grip_cmd = bool(float(np.clip(a[2], 0.0, 1.0)) > 0.5)

        # Gripper transitions: open->closed near object attaches; closed->open releases.
        if grip_cmd and not self.gripper_closed:
            d_obj = float(torch.linalg.vector_norm(self.agent_pos - self.object_pos))
            if d_obj <= self.PICK_RADIUS:
                self.attached = True
                self.object_pos = self.agent_pos.clone()
        elif self.gripper_closed and not grip_cmd:
            self.attached = False
        self.gripper_closed = grip_cmd

        # Yaw integrates first; translation uses the new heading.
        self.agent_yaw = float(
            (self.agent_yaw + yaw_rate * self.DT + np.pi) % (2 * np.pi) - np.pi
        )

        total_mass = self.MASS + (self.OBJECT_MASS if self.attached else 0.0)
        accel = thrust / total_mass
        self.agent_speed = max((self.agent_speed + accel * self.DT) * self.LIN_DAMPING, 0.0)

        heading = np.array(
            [np.cos(self.agent_yaw), np.sin(self.agent_yaw)], dtype=np.float32
        )
        cur = self.agent_pos.detach().cpu().numpy()
        # Push the pre-step position onto the trail before integrating.
        self._trail.append(self.agent_pos.detach().clone())
        nxt = cur + heading * self.agent_speed * self.DT
        clipped = np.clip(nxt, 0.0, 1.0)
        if not np.allclose(clipped, nxt):
            # Damp speed on wall contact rather than zeroing it — at v=0 the min-turning-
            # radius cap (ω≤v/R_MIN) deadlocks the agent against the wall.
            self.agent_speed *= 0.5
        self.agent_pos = torch.tensor(clipped, dtype=self.dtype)

        if self.attached:
            self.object_pos = self.agent_pos.clone()

        # Goal-dwell termination: count consecutive steps with object inside the goal.
        d_goal = float(torch.linalg.vector_norm(self.object_pos - self.goal_pos))
        if d_goal <= self.DROP_RADIUS:
            self.dwell_counter += 1
        else:
            self.dwell_counter = 0
        if self.dwell_counter >= self.GOAL_DWELL_STEPS:
            self.delivered = True

        # Shaped reward.
        if not self.attached:
            d_obj = float(torch.linalg.vector_norm(self.agent_pos - self.object_pos))
            reward = -d_obj
        else:
            reward = -d_goal + 1.0
        if self.dwell_counter > 0:
            reward += 0.5
        if self.delivered:
            reward += 10.0

        self.steps += 1
        terminated = bool(self.delivered)
        truncated = bool(self.steps >= self.MAX_EPISODE_STEPS)

        if self._record:
            self._frames.append(self.current_frame())

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self.current_frame().detach().cpu().numpy()
        return None

    def close(self):
        self._frames = []

    # ---- observation / info ----
    def _get_obs(self) -> np.ndarray:
        ap = self.agent_pos.detach().cpu().numpy()
        op = self.object_pos.detach().cpu().numpy()
        gp = self.goal_pos.detach().cpu().numpy()
        return np.array(
            [
                ap[0], ap[1],
                float(np.cos(self.agent_yaw)), float(np.sin(self.agent_yaw)),
                op[0], op[1],
                gp[0], gp[1],
                1.0 if self.attached else 0.0,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return {
            "attached": bool(self.attached),
            "gripper_closed": bool(self.gripper_closed),
            "delivered": bool(self.delivered),
            "dwell": int(self.dwell_counter),
            "steps": int(self.steps),
            "yaw": float(self.agent_yaw),
            "speed": float(self.agent_speed),
        }

    # ---- rendering ----
    def current_frame(self) -> torch.Tensor:
        frame = self._bg.view(1, 1, 3).expand(self.H, self.W, 3).clone()

        # Goal: hollow circle (annulus).
        gx = float(self.goal_pos[0]); gy = float(self.goal_pos[1])
        gd2 = (self.grid_x - gx) ** 2 + (self.grid_y - gy) ** 2
        gr_outer = self.GOAL_RADIUS
        gr_inner = max(0.0, gr_outer - self.GOAL_OUTLINE)
        goal_mask = (gd2 <= gr_outer ** 2) & (gd2 >= gr_inner ** 2)
        frame[goal_mask] = self._goal_c

        # Agent trail: oldest -> newest, with linearly increasing opacity.
        ar = self.AGENT_RADIUS
        n_trail = len(self._trail)
        if n_trail > 0:
            for i, pos in enumerate(self._trail):
                # i = 0 is oldest; opacity grows toward the present.
                alpha = (i + 1) / (n_trail + 1)
                px = float(pos[0]); py = float(pos[1])
                tmask = (self.grid_x - px) ** 2 + (self.grid_y - py) ** 2 <= ar ** 2
                frame[tmask] = self._agent_c * alpha + frame[tmask] * (1.0 - alpha)

        # Object: filled circle.
        ox = float(self.object_pos[0]); oy = float(self.object_pos[1])
        or_ = self.OBJECT_RADIUS
        obj_mask = (self.grid_x - ox) ** 2 + (self.grid_y - oy) ** 2 <= or_ ** 2
        frame[obj_mask] = self._obj_c

        # Agent: filled circle (current pose, fully opaque on top of the trail).
        ax = float(self.agent_pos[0]); ay = float(self.agent_pos[1])
        agent_mask = (self.grid_x - ax) ** 2 + (self.grid_y - ay) ** 2 <= ar ** 2
        frame[agent_mask] = self._agent_c

        # Gripper-closed indicator: small white inner dot.
        if self.gripper_closed:
            inner_r = ar * 0.45
            inner_mask = (self.grid_x - ax) ** 2 + (self.grid_y - ay) ** 2 <= inner_r ** 2
            frame[inner_mask] = self._grip_c

        # Yaw indicator: black arrow (thick shaft + triangular head) along heading.
        cy = float(np.cos(self.agent_yaw))
        sy = float(np.sin(self.agent_yaw))
        u = (self.grid_x - ax) * cy + (self.grid_y - ay) * sy
        v = -(self.grid_x - ax) * sy + (self.grid_y - ay) * cy
        shaft_len = ar * 2.4
        head_len = ar * 1.25
        shaft_thick = ar * 0.4
        head_base = ar * 1.0
        shaft_mask = (u >= 0) & (u <= shaft_len) & (v.abs() <= shaft_thick)
        # Triangular head from u in [shaft_len, shaft_len + head_len], tapering to a tip.
        head_u = u - shaft_len
        head_mask = (
            (head_u >= 0)
            & (head_u <= head_len)
            & (v.abs() <= head_base * (1.0 - head_u / head_len))
        )
        frame[shaft_mask | head_mask] = self._yaw_c

        # Planned-path overlay (debug): black dot + short arrow at each future waypoint.
        pp = self.planned_path
        if pp is not None and len(pp) > 0:
            pp = np.asarray(pp, dtype=np.float32)
            has_yaw = pp.ndim == 2 and pp.shape[1] >= 3
            dot_r = ar * 0.35
            pp_shaft_len = ar * 1.0
            pp_shaft_thick = ar * 0.12
            for i in range(len(pp)):
                px = float(pp[i, 0]); py = float(pp[i, 1])
                dot_mask = (self.grid_x - px) ** 2 + (self.grid_y - py) ** 2 <= dot_r ** 2
                frame[dot_mask] = self._yaw_c
                if has_yaw:
                    yaw_i = float(pp[i, 2])
                elif i + 1 < len(pp):
                    dx = float(pp[i + 1, 0] - px); dy = float(pp[i + 1, 1] - py)
                    if dx * dx + dy * dy < 1e-12:
                        continue
                    yaw_i = float(np.arctan2(dy, dx))
                else:
                    continue
                cyi = float(np.cos(yaw_i)); syi = float(np.sin(yaw_i))
                pu = (self.grid_x - px) * cyi + (self.grid_y - py) * syi
                pv = -(self.grid_x - px) * syi + (self.grid_y - py) * cyi
                pp_mask = (pu >= 0) & (pu <= pp_shaft_len) & (pv.abs() <= pp_shaft_thick)
                frame[pp_mask] = self._yaw_c

        return frame.clamp(0, 255).to(torch.uint8)

    def render_overlay(self, poses, object_pos=None, goal_pos=None,
                       colors=None, alpha: float = 0.55) -> torch.Tensor:
        """Render a single frame with every (x, y[, yaw]) pose in `poses` superimposed.
        Object/goal default to the env's current state; pass overrides to pin them
        (e.g., the episode's *starting* object_pos, since it follows the agent once
        attached). `colors` is an optional (N, 3) RGB-in-[0,255] array — pass one
        normalized against the same scale (e.g. MAX_EPISODE_STEPS) across rollouts
        if you want time-color to be comparable. Defaults to AGENT_COLOR."""
        frame = self._bg.view(1, 1, 3).expand(self.H, self.W, 3).clone()

        gp = self.goal_pos if goal_pos is None else goal_pos
        gx = float(gp[0]); gy = float(gp[1])
        gd2 = (self.grid_x - gx) ** 2 + (self.grid_y - gy) ** 2
        gr_outer = self.GOAL_RADIUS
        gr_inner = max(0.0, gr_outer - self.GOAL_OUTLINE)
        goal_mask = (gd2 <= gr_outer ** 2) & (gd2 >= gr_inner ** 2)
        frame[goal_mask] = self._goal_c

        op = self.object_pos if object_pos is None else object_pos
        ox = float(op[0]); oy = float(op[1])
        or_ = self.OBJECT_RADIUS
        obj_mask = (self.grid_x - ox) ** 2 + (self.grid_y - oy) ** 2 <= or_ ** 2
        frame[obj_mask] = self._obj_c

        ar = self.AGENT_RADIUS
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim == 1:
            poses = poses.reshape(-1, 2)
        N = len(poses)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32).reshape(N, 3)
        for i in range(N):
            ax, ay = float(poses[i, 0]), float(poses[i, 1])
            c = torch.from_numpy(colors[i]) if colors is not None else self._agent_c
            d2 = (self.grid_x - ax) ** 2 + (self.grid_y - ay) ** 2
            am = d2 <= ar ** 2
            frame[am] = c * alpha + frame[am] * (1.0 - alpha)

        return frame.clamp(0, 255).to(torch.uint8)

    # ---- video / state helpers ----
    def save_video(self, path: str):
        fps = max(1, int(round(1.0 / self.DT)))
        assert len(self._frames) > 0, "No frames recorded. Set env._record = True before reset()."
        frames = torch.stack(self._frames, dim=0).to(torch.uint8).cpu().numpy()
        T, H, W, _ = frames.shape

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "save_video requires the `ffmpeg` binary on PATH (e.g. `apt install ffmpeg`)."
            )
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = proc.communicate(frames.tobytes())
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {err.decode(errors='replace')}")

        sidecar_path = os.path.splitext(path)[0] + ".json"
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump({"video_path": path, "fps": int(fps)}, fh, indent=2)

        self._frames = []

    def get_env_state(self) -> dict:
        return {
            "agent_pos": self.agent_pos.detach().cpu().clone(),
            "agent_yaw": float(self.agent_yaw),
            "agent_speed": float(self.agent_speed),
            "object_pos": self.object_pos.detach().cpu().clone(),
            "goal_pos": self.goal_pos.detach().cpu().clone(),
            "attached": bool(self.attached),
            "gripper_closed": bool(self.gripper_closed),
            "dwell_counter": int(self.dwell_counter),
            "delivered": bool(self.delivered),
            "steps": int(self.steps),
        }

    def set_env_state(self, s: dict):
        self.agent_pos = s["agent_pos"].to(dtype=self.dtype)
        self.agent_yaw = float(s["agent_yaw"])
        self.agent_speed = float(s["agent_speed"])
        self.object_pos = s["object_pos"].to(dtype=self.dtype)
        self.goal_pos = s["goal_pos"].to(dtype=self.dtype)
        self.attached = bool(s["attached"])
        self.gripper_closed = bool(s["gripper_closed"])
        self.dwell_counter = int(s["dwell_counter"])
        self.delivered = bool(s["delivered"])
        self.steps = int(s["steps"])
