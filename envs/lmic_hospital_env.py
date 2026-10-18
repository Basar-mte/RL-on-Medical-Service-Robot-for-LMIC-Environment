"""
LMICHospitalNav-v1 — Gymnasium RL environment.

Composes FloorPlan + PedestrianCrowd + an Ackermann-steered (bicycle-model)
robot into a fully-specified, check_env-compliant Gymnasium environment.
Sensor suite and kinematics are modeled after the target hardware platform
(Yahboom A1 Ackermann chassis + LDROBOT LD19 LiDAR + pan-tilt monocular
camera — see WHEELBASE/MAX_STEER/CAMERA_* constants below).

Observation (gymnasium.spaces.Dict of float32/uint8 arrays)
-------------------------------------------------------------
lidar       (48,)          – LiDAR ray distances, normalised [0,1] over
                              [0, LIDAR_MAX] m, LIDAR_FOV=360° field of view
robot_state  (5,)          – [x_n, y_n, sin(θ), cos(θ), v_n]
pedestrians (20,)          – 5 nearest pedestrians, each (dx, dy, vx, vy),
                              normalised by PED_NEIGHBOR_RADIUS / PED_MAX_SPEED
context     (14,)          – [density_5m, task_urgency, zone_onehot×8,
                              sin(tod), cos(tod), density_2m,
                              cultural_zone_distance/10]
camera      (84,84,3) u8   – synthetic egocentric raster (forward-facing,
                              occlusion-aware crop of the same 2D scene the
                              pygame renderer draws) — see _render_camera().
                              NOT a photorealistic render; a stylized proxy
                              for a real onboard monocular camera. Excluded
                              from VecNormalize's obs normalisation.

Action  (2,) — Ackermann / bicycle-model steering
---------------------------------------------------
[v, δ]  v ∈ [MIN_LINEAR, MAX_LINEAR] m/s (rear-axle drive speed; v<0 = reverse),
        δ ∈ [-MAX_STEER, MAX_STEER] rad (front steering angle)
Heading rate = (v / WHEELBASE) * tan(δ) — heading cannot change at v=0,
matching a real Ackermann-steered vehicle (no in-place pivot).

Info dict keys (always present)
---------------------------------
collision, task_complete, steps, social_violation, cultural_violation,
current_zone, dist_to_goal, min_pedestrian_distance, context_vector,
path_length_optimal, task_id, urgency, density_level
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.floorplan import FloorPlan, load_floorplan
from envs.pedestrians import PedestrianCrowd

# -------------------------------------------------------------------
# Task catalogue (mirrors playbook §3.2)
# -------------------------------------------------------------------
TASKS = {
    "STAT_medicine": dict(urgency=1.0, deadline_s=45, label="STAT Medicine"),
    "nurse_call":    dict(urgency=0.9, deadline_s=60, label="Nurse Call"),
    "specimen":      dict(urgency=0.6, deadline_s=90, label="Specimen Transport"),
    "restock":       dict(urgency=0.3, deadline_s=180, label="Routine Restock"),
    "idle_return":   dict(urgency=0.1, deadline_s=300, label="Return to Base"),
}
TASK_IDS = list(TASKS.keys())

# -------------------------------------------------------------------
# Density levels — calibrated to LMIC hospital reality (DMCH-scale)
# LMIC hospitals have 2-4× the pedestrian density of Western benchmarks
# Medium reflects typical DMCH corridor occupancy at 133% bed capacity
# -------------------------------------------------------------------
DENSITY_LEVELS = {
    "low":    dict(medical_staff=4,  patient=8,  family_attendant=8,  wheelchair=3),
    "medium": dict(medical_staff=8,  patient=16, family_attendant=16, wheelchair=5),
    "high":   dict(medical_staff=14, patient=26, family_attendant=26, wheelchair=8),
}

# Robot parameters
ROBOT_RADIUS  = 0.25   # m
MAX_LINEAR    = 0.8    # m/s
# Reverse: the physical Yahboom A1 chassis can back up, and without it a
# no-pivot Ackermann robot facing an obstacle at spawn has no recovering
# action at all (an irreducible, policy-independent failure floor).
MIN_LINEAR    = -0.3   # m/s
DT            = 0.1    # s
MAX_STEPS     = 1000
N_LIDAR_RAYS  = 48
LIDAR_MAX     = 12.0   # m   (LDROBOT LD19 spec)
LIDAR_FOV     = 360.0  # degrees (LDROBOT LD19 spec)
K_NEIGHBORS   = 5
PED_NEIGHBOR_RADIUS = 5.0   # m — matches PedestrianCrowd.get_local_neighbors default
PED_MAX_SPEED       = 2.5   # m/s — safety margin over max desired_speed*MAX_SPEED_FACTOR (~2.1 m/s)

# Ackermann / bicycle-model steering — target hardware: Yahboom A1 Ackermann
# chassis (rear-wheel drive + front steering servo). WHEELBASE and MAX_STEER
# are PLACEHOLDERS: the real chassis's exact wheelbase/steering-servo throw
# are not published anywhere (checked Yahboom's site/GitHub/RobotShop) — they
# must be measured on the physical robot and updated here before real-hardware
# deployment. Implied min turning radius at these placeholders: L/tan(MAX_STEER)
# ≈ 0.35 m, a plausible order of magnitude for a small chassis, not a spec claim.
WHEELBASE     = 0.20              # m
MAX_STEER     = math.radians(30)  # rad

# Camera sensor — synthetic egocentric raster (see _render_camera()); this is
# a 2D grid-world sim with no 3D renderer, so this is a stylized proxy for a
# real forward-facing monocular camera, not a photorealistic simulation.
CAMERA_FOV_DEG  = 90.0   # degrees, forward-facing (narrower than the 360° LiDAR)
CAMERA_RANGE_M  = 5.0    # m
CAMERA_IMG_SIZE = 84     # px, matches SB3 NatureCNN's traditional 84x84 convention
CAMERA_CHANNELS = 3

# Shared floor/pedestrian color LUTs — single source of truth for _draw_floor,
# _render_rgb, _draw_pedestrians, and _render_camera (previously two
# independently-maintained, inconsistent ZONE_COLORS dicts existed here).
ZONE_COLORS = {
    0:  (80,  80,  80),   # wall
    1:  (220, 220, 220),  # walkable
    2:  (200, 200, 160),  # corridor
    3:  (160, 220, 160),  # nurse
    4:  (160, 200, 220),  # pharmacy
    5:  (220, 210, 160),  # ward
    6:  (200, 160, 220),  # icu
    7:  (220, 160, 160),  # emergency
    8:  (130, 190, 130),  # prayer
    9:  (160, 200, 220),  # male ward
    10: (220, 200, 180),  # female ward
}
PCLASS_COLORS = {
    "medical_staff":    (0, 100, 200),
    "patient":          (200, 100, 0),
    "family_attendant": (180, 50, 180),
    "wheelchair":       (50, 180, 50),
}


def lidar_ray_angles(fov_deg: float, n_rays: int) -> np.ndarray:
    """
    Ray bearing offsets (rad) relative to robot heading, evenly spanning
    `fov_deg`. At exactly 360 degrees, `endpoint=False` avoids placing a
    duplicate ray at -180/+180 (the same physical direction); for any FOV
    below 360, `endpoint=True` is correct since the two extremes are
    physically distinct directions.

    Shared by envs.lmic_hospital_env._cast_lidar and
    baselines.classical.SocialForceController so the two can never silently
    diverge on ray geometry (importing only the FOV/ray-count constants
    would still allow the `endpoint` argument itself to drift).
    """
    return np.linspace(
        -math.radians(fov_deg / 2), math.radians(fov_deg / 2),
        n_rays, endpoint=(fov_deg < 360.0),
    )


class LMICHospitalNavEnv(gym.Env):
    """
    LMICHospitalNav-v1 Gymnasium environment.

    Parameters
    ----------
    floorplan_path : str or Path
        Path to the YAML floor-plan config.
    reward_fn : callable, optional
        reward_fn(obs, next_obs, action, info) -> float.
        If None, uses the default CARS reward with equal weights.
    weight_source : callable, optional
        weight_source(context_vec) -> np.ndarray (4,).
        If None, uses fixed equal weights [0.25, 0.25, 0.25, 0.25].
    render_mode : str, optional
        'human' or 'rgb_array'.
    seed : int, optional
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        floorplan_path: Optional[str | Path] = None,
        reward_fn: Optional[Callable] = None,
        weight_source: Optional[Callable] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        use_camera: bool = True,
    ) -> None:
        super().__init__()

        if floorplan_path is None:
            floorplan_path = Path(__file__).parent.parent / "configs" / "dmch_hospital.yaml"
        self.floorplan_path = Path(floorplan_path)
        self.fp = load_floorplan(self.floorplan_path)
        self.render_mode = render_mode
        # use_camera=False drops the synthetic camera sensor entirely (no
        # "camera" obs key, no per-step render): the policy then trains on
        # the 85-dim vector portion alone, which removes the NatureCNN from
        # every SAC gradient step — the dominant wall-clock cost.
        self.use_camera = use_camera

        # Injected reward / weight source
        self._reward_fn = reward_fn
        self._weight_source = weight_source

        # Spaces — Dict observation: LiDAR + robot state + pedestrians +
        # context (each properly bounded per-key) + a synthetic camera image
        # (unless use_camera=False).
        obs_spaces = {
            "lidar":       spaces.Box(0.0, 1.0, shape=(N_LIDAR_RAYS,), dtype=np.float32),
            "robot_state": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
            "pedestrians": spaces.Box(-1.0, 1.0, shape=(K_NEIGHBORS * 4,), dtype=np.float32),
            "context":     spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32),
            # Task-goal features: [sin(bearing err), cos(bearing err),
            # dist/GOAL_DIST_NORM clipped to 1]. Without these the observation
            # carried NO goal information at all once the camera (whose
            # rendered goal marker was the only goal signal) is disabled —
            # the policy could only infer the destination from the task's
            # urgency scalar, which made goal-reaching essentially
            # unlearnable (measured: SR pinned at exactly 0 through 80k
            # steps while survival time tripled).
            "goal":        spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        }
        if self.use_camera:
            obs_spaces["camera"] = spaces.Box(
                0, 255, shape=(CAMERA_IMG_SIZE, CAMERA_IMG_SIZE, CAMERA_CHANNELS), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)
        # Ackermann/bicycle-model action: [v, steering_angle]; v < 0 = reverse
        self.action_space = spaces.Box(
            low=np.array([MIN_LINEAR, -MAX_STEER], dtype=np.float32),
            high=np.array([MAX_LINEAR, MAX_STEER], dtype=np.float32),
        )

        # Internal state
        self._rng = np.random.default_rng(seed)
        self._crowd = PedestrianCrowd(self.fp, max_pedestrians=80, rng=self._rng)
        self._reset_state()

        # Pygame renderer
        self._screen = None
        self._clock = None
        self._cell_px = 16  # pixels per cell at 0.1 m
        # Cached static floor/zone color raster (see _floor_color_grid) —
        # zone_map/occupancy never change after FloorPlan.__init__, so the
        # per-step camera render samples this instead of re-drawing cells.
        self._floor_rgb: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._crowd = PedestrianCrowd(self.fp, max_pedestrians=80, rng=self._rng)

        opts = options or {}
        task_id = opts.get("task", TASK_IDS[self._rng.integers(len(TASK_IDS))])
        # Density persists across auto-resets: SB3's VecEnvs call reset()
        # without options after every episode, so a hardcoded "medium"
        # default silently reverted any non-medium training/eval run to
        # medium from episode 2 onward.
        density_key = opts.get("density", getattr(self, "_density_key", "medium"))

        # Stateful injected reward functions (He2025RewardFn/VanillaSACReward
        # carry _prev_dist) must be reset per episode — nothing ever called
        # their reset() before, so the first step of every episode computed
        # progress against the PREVIOUS episode's final goal distance.
        if self._reward_fn is not None and hasattr(self._reward_fn, "reset"):
            self._reward_fn.reset()

        self._setup_episode(task_id, density_key)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v, steering_angle = float(action[0]), float(action[1])

        # Kinematic update — Ackermann/bicycle model (rear-axle reference).
        # heading_rate is exactly zero at v=0 for any steering_angle: unlike
        # a differential-drive unicycle model, this robot cannot pivot in
        # place, matching the real Ackermann-steered chassis.
        heading_rate = (v / WHEELBASE) * math.tan(steering_angle)
        self._robot_heading += heading_rate * DT
        self._robot_heading = (self._robot_heading + math.pi) % (2 * math.pi) - math.pi
        dx = v * math.cos(self._robot_heading) * DT
        dy = v * math.sin(self._robot_heading) * DT
        new_x = self._robot_pos[0] + dx
        new_y = self._robot_pos[1] + dy

        # Collision check — against the radius-inflated grid: the previous
        # raw-grid point test ignored ROBOT_RADIUS entirely, letting the
        # 0.5 m-wide body overlap beds/walls by up to 0.25 m unregistered
        # (paper eq. (2): collision iff d(p, O) <= r_r).
        collision = False
        if not self.fp.is_walkable_robot(new_x, new_y):
            collision = True
            new_x, new_y = self._robot_pos[0], self._robot_pos[1]
            v = 0.0

        # Pedestrian collision
        for agent in self._crowd.agents:
            dist = math.hypot(agent.pos[0] - new_x, agent.pos[1] - new_y)
            if dist < (ROBOT_RADIUS + agent.radius):
                collision = True
                break

        self._robot_pos = np.array([new_x, new_y])
        self._robot_speed = v
        self._step += 1
        self._time += DT

        # Advance crowd
        self._crowd.step(DT, robot_pos=self._robot_pos)

        # Time-of-day (simulated 24h cycle, sped up × factor)
        self._tod_hour = (self._tod_hour + DT / 3600 * 48) % 24  # 48× speed

        # Check goal. Collision takes precedence: previously a step that
        # entered the goal radius while ramming a pedestrian scored (and was
        # rewarded) as a clean success — goal bonus + task bonus outweighed
        # the collision penalty, so clipping pedestrians on final approach
        # was reward-dominant and such collisions vanished from the metrics.
        dist_to_goal = float(np.linalg.norm(self._robot_pos - self._task_goal))
        task_complete = bool(dist_to_goal < 0.6 and not collision)

        # Compute reward
        obs = self._get_obs()
        info = self._get_info(collision=collision, task_complete=task_complete)

        if self._reward_fn is not None:
            reward = self._reward_fn(self._prev_obs, obs, action, info)
        else:
            reward = self._default_reward(collision, task_complete, action, info)

        self._prev_obs = obs
        self._prev_dist = dist_to_goal

        terminated = bool(collision or task_complete)
        truncated = bool(self._step >= MAX_STEPS)

        if self.render_mode == "human":
            self._render_human()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._screen = None

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

    def _setup_episode(self, task_id: str, density_key: str) -> None:
        self._task_id = task_id
        self._density_key = density_key
        task_cfg = TASKS[task_id]
        self._urgency = task_cfg["urgency"]
        self._deadline = task_cfg["deadline_s"]

        # Robot spawn at nurse station area (snapped to robot-walkable space)
        spawn_pts = self.fp.robot_spawn_points
        spawn = spawn_pts[self._rng.integers(len(spawn_pts))]
        self._robot_pos = self.fp.nearest_walkable(
            spawn[0], spawn[1], grid=self.fp.occupancy_inflated)
        self._robot_heading = self._rng.uniform(0, 2 * math.pi)
        self._robot_speed = 0.0

        # Task goal from floor plan config
        task_def = self.fp.tasks.get(task_id, {})
        target_room = task_def.get("target_room_id", "emergency_dept")
        try:
            goal_center = self.fp.room_center(target_room)
        except KeyError:
            goal_center = self.fp.random_walkable_point(self._rng)
        # Snap onto the ROBOT-walkable (radius-inflated) grid so every task
        # goal is a position the chassis can physically occupy.
        self._task_goal = self.fp.nearest_walkable(
            goal_center[0], goal_center[1], grid=self.fp.occupancy_inflated)

        # Guard against spawn-inside-goal: idle_return's base is the nurse
        # station — the robot's own spawn area — which previously produced
        # 1-step "successes" requiring no navigation at all (they were the
        # entirety of every agent's measured success rate) and leaked free
        # task bonuses into training. Require a minimum spawn->goal distance,
        # re-picking the spawn point among the fixed candidates if needed.
        MIN_SPAWN_GOAL = 3.0
        if np.linalg.norm(self._robot_pos - self._task_goal) < MIN_SPAWN_GOAL:
            far = [p for p in spawn_pts
                   if np.linalg.norm(np.asarray(p, dtype=float) - self._task_goal)
                   >= MIN_SPAWN_GOAL]
            if far:
                self._robot_pos = np.array(far[self._rng.integers(len(far))], dtype=float)
            else:
                self._robot_pos = np.array(
                    max(spawn_pts, key=lambda p: np.linalg.norm(
                        np.asarray(p, dtype=float) - self._task_goal)), dtype=float)

        # Pedestrian crowd — never spawned inside the robot's immediate
        # footprint (instant, policy-independent spawn collisions otherwise)
        counts = DENSITY_LEVELS.get(density_key, DENSITY_LEVELS["medium"])
        self._crowd.reset(counts, rng=np.random.default_rng(int(self._rng.integers(1e6))),
                          avoid_pos=self._robot_pos, avoid_radius=1.5)

        # Timing
        self._step = 0
        self._time = 0.0
        self._tod_hour = float(self._rng.uniform(6.0, 22.0))
        self._prev_dist = np.linalg.norm(self._robot_pos - self._task_goal)
        self._optimal_path_len = self.fp.path_length(self._robot_pos, self._task_goal)
        self._total_path_len = 0.0
        self._prev_pos = self._robot_pos.copy()
        self._cultural_violations = 0
        self._social_violations = 0
        self._ep_component_sums = np.zeros(4, dtype=float)
        self._prev_obs = self._get_obs()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = {
            "lidar": self._cast_lidar(),
            "robot_state": self._robot_state_vec(),
            "pedestrians": self._ped_obs_vec(),
            "context": self._context_vec(),
            "goal": self._goal_vec(),
        }
        if self.use_camera:
            obs["camera"] = self._render_camera()
        for key, subspace in self.observation_space.spaces.items():
            if key == "camera":
                continue
            obs[key] = np.clip(obs[key], subspace.low, subspace.high)
        return obs

    def _ped_obs_vec(self) -> np.ndarray:
        """
        5 nearest pedestrians' (dx, dy, vx, vy), properly normalised to
        [-1, 1] by PED_NEIGHBOR_RADIUS / PED_MAX_SPEED before being bounded.
        (Previously these raw metre/m-per-s values were clipped against a
        blanket [-1, 1] Box meant for a different obs slice, silently
        truncating any pedestrian more than ~1m away to exactly +-1.0.)
        """
        raw = self._crowd.get_local_neighbors(
            self._robot_pos, k=K_NEIGHBORS, radius=PED_NEIGHBOR_RADIUS
        )  # (K, 4): dx, dy, vx, vy
        norm = raw.copy()
        norm[:, 0:2] /= PED_NEIGHBOR_RADIUS
        norm[:, 2:4] /= PED_MAX_SPEED
        return norm.flatten().astype(np.float32)

    def _cast_lidar(self) -> np.ndarray:
        """
        Vectorized LiDAR: all rays sampled simultaneously via numpy.
        ~20x faster than sequential Python loop.
        """
        step = self.fp.cell_size          # 0.1 m per sample
        n_samples = int(LIDAR_MAX / step)  # 100 samples per ray

        # Ray directions: (N_RAYS,)
        base_angles = lidar_ray_angles(LIDAR_FOV, N_LIDAR_RAYS)
        world_angles = self._robot_heading + base_angles
        dx = np.cos(world_angles)   # (N_RAYS,)
        dy = np.sin(world_angles)   # (N_RAYS,)

        # Sample distances along each ray: (n_samples,)
        dists = np.arange(1, n_samples + 1, dtype=float) * step

        # All sample world coords: (N_RAYS, n_samples)
        xs = self._robot_pos[0] + dx[:, None] * dists[None, :]
        ys = self._robot_pos[1] + dy[:, None] * dists[None, :]

        # Out-of-bounds mask
        oob = (xs < 0) | (ys < 0) | (xs >= self.fp.width_m) | (ys >= self.fp.height_m)

        # Grid indices (clipped to avoid out-of-bounds array access)
        cols = np.clip((xs / self.fp.cell_size).astype(int), 0, self.fp.width_cells - 1)
        rows = np.clip((ys / self.fp.cell_size).astype(int), 0, self.fp.height_cells - 1)

        # Wall hits: (N_RAYS, n_samples)
        wall_hit = self.fp.occupancy[rows, cols].astype(bool) | oob

        # First wall-blocked sample per ray (argmax on bool: first True)
        first_hit_idx = np.argmax(wall_hit, axis=1)       # (N_RAYS,)
        any_hit       = wall_hit.any(axis=1)              # (N_RAYS,)
        wall_dist = np.where(any_hit, dists[first_hit_idx], LIDAR_MAX)

        # Pedestrian hits: exact ray-circle intersection, (N_RAYS, P) —
        # replaces the (N_RAYS x n_samples x P) ~260k-element broadcast that
        # profiled at ~63% of env.step wall time, and is exact rather than
        # quantized to the 0.1 m sampling grid.
        if self._crowd.agents:
            ped_pos = np.array([a.pos for a in self._crowd.agents])    # (P, 2)
            ped_rad = np.array([a.radius for a in self._crowd.agents])  # (P,)
            rel = ped_pos - self._robot_pos[None, :]                    # (P, 2)
            # Projection of each ped centre onto each ray direction
            t_ctr = dx[:, None] * rel[None, :, 0] + dy[:, None] * rel[None, :, 1]  # (R, P)
            perp2 = (rel[None, :, 0] - t_ctr * dx[:, None]) ** 2 \
                  + (rel[None, :, 1] - t_ctr * dy[:, None]) ** 2                  # (R, P)
            r2 = (ped_rad ** 2)[None, :]
            hit = (perp2 <= r2) & (t_ctr >= 0.0)
            half_chord = np.sqrt(np.maximum(r2 - perp2, 0.0))
            t_hit = np.where(hit, np.maximum(t_ctr - half_chord, 0.0), LIDAR_MAX)
            ped_dist = t_hit.min(axis=1)                                # (R,)
        else:
            ped_dist = np.full(N_LIDAR_RAYS, LIDAR_MAX)

        hits = np.minimum(wall_dist, ped_dist)
        hits = np.minimum(hits, LIDAR_MAX).astype(np.float32)
        return hits / LIDAR_MAX

    def _robot_state_vec(self) -> np.ndarray:
        x_n = (self._robot_pos[0] / self.fp.width_m) * 2 - 1
        y_n = (self._robot_pos[1] / self.fp.height_m) * 2 - 1
        v_n = self._robot_speed / MAX_LINEAR
        return np.array([x_n, y_n, math.sin(self._robot_heading),
                         math.cos(self._robot_heading), v_n], dtype=np.float32)

    GOAL_DIST_NORM = 25.0  # m; distances beyond this read as 1.0

    def _goal_vec(self) -> np.ndarray:
        """[sin, cos of goal bearing in the ROBOT frame, normalised distance]."""
        delta = self._task_goal - self._robot_pos
        bearing = math.atan2(delta[1], delta[0]) - self._robot_heading
        dist_n = min(float(np.linalg.norm(delta)) / self.GOAL_DIST_NORM, 1.0)
        return np.array([math.sin(bearing), math.cos(bearing), dist_n],
                        dtype=np.float32)

    def _context_vec(self) -> np.ndarray:
        # Unsaturated densities: the old crowd_density normalised by 10
        # pedestrians-within-5m and clipped, so at LMIC densities (45-74
        # agents) it read a constant 1.0 — a weight network cannot adapt to
        # a signal that never varies. /30 within 5 m and /8 within 2 m keep
        # the full operating range informative.
        dists = np.array([np.linalg.norm(a.pos - self._robot_pos)
                          for a in self._crowd.agents]) if self._crowd.agents else np.array([])
        density5 = min(float((dists <= 5.0).sum()) / 30.0, 1.0) if dists.size else 0.0
        density2 = min(float((dists <= 2.0).sum()) / 8.0, 1.0) if dists.size else 0.0
        zone_oh = self.fp.zone_onehot(self._robot_pos[0], self._robot_pos[1])
        tod_sin = math.sin(2 * math.pi * self._tod_hour / 24)
        tod_cos = math.cos(2 * math.pi * self._tod_hour / 24)
        # Distance to the nearest cultural zone (0 inside one, capped 10 m):
        # lets a weight network raise the cultural weight BEFORE crossing a
        # threshold rather than only after (anticipation, paper Fig. 6).
        cult_d = self.fp.cultural_zone_distance(
            self._robot_pos[0], self._robot_pos[1]) / 10.0
        ctx = np.concatenate([
            [density5, self._urgency],
            zone_oh,
            [tod_sin, tod_cos],
            [density2, cult_d],
        ]).astype(np.float32)
        return ctx

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _default_reward(
        self, collision: bool, task_complete: bool,
        action: np.ndarray, info: Dict
    ) -> float:
        """
        Default CARS reward with fixed equal weights.
        Weights: [w_nav=0.25, w_safety=0.25, w_social=0.25, w_task=0.25]
        """
        from rewards.components import (
            navigation_reward, safety_reward, task_reward
        )
        from rewards.cultural import cultural_reward

        dist = info["dist_to_goal"]
        r_nav = navigation_reward(self._prev_dist, dist, task_complete, action)
        r_safe = safety_reward(collision, info["min_pedestrian_distance"])
        r_cult, cv = cultural_reward(
            self._robot_pos, self.fp, self._tod_hour,
            self._crowd.nearest_family_distance(self._robot_pos)
        )
        r_task = task_reward(task_complete, self._urgency, self._time, self._deadline)

        # Use CWN weight source if available, else equal weights
        ctx = self._context_vec()
        if self._weight_source is not None:
            w = self._weight_source(ctx)
        else:
            w = np.array([0.25, 0.25, 0.25, 0.25])
        total = (w[0] * r_nav + w[1] * r_safe + w[2] * r_cult + w[3] * r_task)

        # Unweighted per-step component sums, exposed via info at episode end
        # — the CWN/attention REINFORCE credit uses the EQUAL-WEIGHT return
        # built from these (see train.callbacks.CWNCallback), so the weight
        # network cannot game its own scalarization. NOTE: violation counting
        # moved to _get_info (single authoritative place, all agents).
        self._ep_component_sums += np.array([r_nav, r_safe, r_cult, r_task])
        return float(np.clip(total, -5.0, 5.0))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self, collision: bool = False, task_complete: bool = False) -> Dict:
        dist = float(np.linalg.norm(self._robot_pos - self._task_goal))
        min_ped = min(
            (np.linalg.norm(a.pos - self._robot_pos) for a in self._crowd.agents),
            default=10.0,
        )
        # Personal-space threshold d_p = 0.8 m (paper eq. (9)). The previous
        # 0.5 m threshold was exactly the collision-termination distance, so
        # "social violation" could only ever fire on the terminal step.
        social_viol = min_ped < 0.8

        zone = self.fp.get_zone(self._robot_pos[0], self._robot_pos[1])
        cult_zone = self.fp.is_cultural_zone(self._robot_pos[0], self._robot_pos[1])
        # Cultural-violation flag per paper eq. (3): presence in a cultural
        # zone OR attendant hard-intrusion (d_fam < d_a = 0.6 m). Counted
        # here — the single authoritative place, on every step, for every
        # agent — previously it was incremented inside _default_reward, so
        # agents with injected reward functions (vanilla, he2025) logged
        # structurally-zero training CVR, and the counter lagged the info
        # dict by one step.
        min_fam = self._crowd.nearest_family_distance(self._robot_pos)
        cult_viol = bool(cult_zone or min_fam < 0.6)
        if self._step > 0:  # don't count the reset()-time info call
            if social_viol:
                self._social_violations += 1
            if cult_viol:
                self._cultural_violations += 1

        # Path length tracking
        move = np.linalg.norm(self._robot_pos - self._prev_pos)
        self._total_path_len += move
        self._prev_pos = self._robot_pos.copy()

        return dict(
            collision=collision,
            task_complete=task_complete,
            # SB3's EvalCallback and Monitor recognise "is_success": exposes
            # training-time success-rate curves in TensorBoard (eval/success_rate).
            is_success=bool(task_complete),
            steps=self._step,
            social_violation=social_viol,
            cultural_violation=cult_viol,
            current_zone=zone,
            episode_component_sums=self._ep_component_sums.copy(),
            dist_to_goal=dist,
            min_pedestrian_distance=float(min_ped),
            context_vector=self._context_vec(),
            path_length_optimal=self._optimal_path_len,
            path_length_actual=self._total_path_len,
            task_id=self._task_id,
            urgency=self._urgency,
            density_level=self._density_key,
            tod_hour=self._tod_hour,
            cultural_violations=self._cultural_violations,
            social_violations=self._social_violations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._robot_pos = np.array([20.0, 13.5])
        self._robot_heading = 0.0
        self._robot_speed = 0.0
        self._task_id = "idle_return"
        self._urgency = 0.1
        self._deadline = 300
        self._task_goal = np.array([20.0, 13.5])
        self._step = 0
        self._time = 0.0
        self._tod_hour = 12.0
        self._prev_dist = 0.0
        self._optimal_path_len = 0.0
        self._total_path_len = 0.0
        self._prev_pos = self._robot_pos.copy()
        self._cultural_violations = 0
        self._social_violations = 0
        self._ep_component_sums = np.zeros(4, dtype=float)  # [nav, safe, cult, task]
        self._density_key = "medium"
        self._prev_obs = {
            key: np.zeros(subspace.shape, dtype=subspace.dtype)
            for key, subspace in self.observation_space.spaces.items()
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_human(self):
        try:
            import pygame
        except ImportError:
            return None

        W = self.fp.width_cells * self._cell_px
        H = self.fp.height_cells * self._cell_px

        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("LMICHospitalNav-v1")
            self._screen = pygame.display.set_mode((W, H + 80))
            self._clock = pygame.time.Clock()

        self._screen.fill((240, 240, 240))
        self._draw_floor()
        self._draw_pedestrians()
        self._draw_goal()
        self._draw_robot()
        self._draw_hud()

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def _render_rgb(self) -> np.ndarray:
        try:
            import pygame
        except ImportError:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        W = self.fp.width_cells * self._cell_px
        H = self.fp.height_cells * self._cell_px

        surface = pygame.Surface((W, H))
        surface.fill((240, 240, 240))

        # Floor/zone cells — vectorized: each cell was drawn as an exact
        # _cell_px-square rect on a _cell_px grid, so a nearest-neighbour
        # upscale of the cached color raster is pixel-identical to the old
        # ~120k-call pygame.draw.rect double loop.
        floor_img = np.repeat(np.repeat(self._floor_color_grid(), self._cell_px, axis=0),
                              self._cell_px, axis=1)
        pygame.surfarray.blit_array(surface, floor_img.swapaxes(0, 1))

        # Pedestrians, goal, robot — previously missing from rgb_array mode
        # (it only rasterized the floor), a real gap vs. human-mode render.
        for agent in self._crowd.agents:
            px, py = self._world_to_px(agent.pos[0], agent.pos[1])
            color = PCLASS_COLORS.get(agent.pclass, (100, 100, 100))
            r_px = max(3, int(agent.radius / self.fp.cell_size * self._cell_px))
            pygame.draw.circle(surface, color, (px, py), r_px)

        gx, gy = self._world_to_px(self._task_goal[0], self._task_goal[1])
        pygame.draw.circle(surface, (255, 200, 0), (gx, gy), 8)
        pygame.draw.circle(surface, (0, 0, 0), (gx, gy), 8, 2)

        rx, ry = self._world_to_px(self._robot_pos[0], self._robot_pos[1])
        r_px = max(4, int(ROBOT_RADIUS / self.fp.cell_size * self._cell_px))
        pygame.draw.circle(surface, (255, 60, 60), (rx, ry), r_px)
        hx = rx + int(r_px * 1.5 * math.cos(self._robot_heading))
        hy = ry + int(r_px * 1.5 * math.sin(self._robot_heading))
        pygame.draw.line(surface, (0, 0, 0), (rx, ry), (hx, hy), 2)

        return np.transpose(
            np.array(pygame.surfarray.array3d(surface), dtype=np.uint8),
            axes=(1, 0, 2),
        )

    def _render_camera(self) -> np.ndarray:
        """
        Synthetic egocentric camera: a small (CAMERA_IMG_SIZE x CAMERA_IMG_SIZE)
        forward-facing, rotation-aligned, occlusion-consistent raster crop of
        the same 2D scene the pygame renderer draws (floor/zones, pedestrians,
        goal), centered on and rotated to the robot's heading, cropped to the
        forward CAMERA_FOV_DEG cone.

        This is a stylized proxy for a real onboard monocular camera — this
        sim has no 3D renderer, so there is no photorealistic ground truth to
        match. Rotation/crop sign conventions are verified via a visual
        smoke test (dumped PNG frames), not assumed correct from inspection
        alone.
        """
        try:
            import pygame
        except ImportError:
            return np.zeros((CAMERA_IMG_SIZE, CAMERA_IMG_SIZE, CAMERA_CHANNELS), dtype=np.uint8)

        px_per_m = 20
        win_px = int(2 * CAMERA_RANGE_M * px_per_m)

        surface = pygame.Surface((win_px, win_px))
        surface.fill((240, 240, 240))

        rx, ry = float(self._robot_pos[0]), float(self._robot_pos[1])
        x0, y0 = rx - CAMERA_RANGE_M, ry - CAMERA_RANGE_M

        def local_to_px(wx: float, wy: float) -> Tuple[int, int]:
            return int((wx - x0) * px_per_m), int((wy - y0) * px_per_m)

        # Floor/zone cells: sample the cached full-map color raster at every
        # window pixel's world coordinate (vectorized nearest-neighbour
        # gather). Replaces the previous per-cell double loop of
        # pygame.draw.rect calls (~10k per frame, the dominant per-step
        # cost); exact per-pixel sampling also needs no +1px overdraw to
        # cover rounding seams — there are none by construction.
        floor_rgb = self._floor_color_grid()
        wx = x0 + (np.arange(win_px) + 0.5) / px_per_m
        wy = y0 + (np.arange(win_px) + 0.5) / px_per_m
        cols = np.floor(wx / self.fp.cell_size).astype(np.int64)
        rows = np.floor(wy / self.fp.cell_size).astype(np.int64)
        in_x = (cols >= 0) & (cols < self.fp.width_cells)
        in_y = (rows >= 0) & (rows < self.fp.height_cells)
        img = np.full((win_px, win_px, CAMERA_CHANNELS), 240, dtype=np.uint8)
        sub = floor_rgb[np.ix_(np.clip(rows, 0, self.fp.height_cells - 1),
                               np.clip(cols, 0, self.fp.width_cells - 1))]
        mask = in_y[:, None] & in_x[None, :]
        img[mask] = sub[mask]
        pygame.surfarray.blit_array(surface, img.swapaxes(0, 1))

        # Pedestrians within window
        for agent in self._crowd.agents:
            if abs(agent.pos[0] - rx) > CAMERA_RANGE_M or abs(agent.pos[1] - ry) > CAMERA_RANGE_M:
                continue
            px, py = local_to_px(agent.pos[0], agent.pos[1])
            color = PCLASS_COLORS.get(agent.pclass, (100, 100, 100))
            r_px = max(2, int(agent.radius * px_per_m))
            pygame.draw.circle(surface, color, (px, py), r_px)

        # Goal marker, if within window
        if abs(self._task_goal[0] - rx) <= CAMERA_RANGE_M and abs(self._task_goal[1] - ry) <= CAMERA_RANGE_M:
            gx, gy = local_to_px(self._task_goal[0], self._task_goal[1])
            pygame.draw.circle(surface, (255, 200, 0), (gx, gy), max(3, int(0.15 * px_per_m)))

        # Robot marker at the window center (world-aligned, pre-rotation)
        ctr = (win_px // 2, win_px // 2)
        r_px = max(3, int(ROBOT_RADIUS * px_per_m))
        pygame.draw.circle(surface, (255, 60, 60), ctr, r_px)

        # Rotate so the robot's forward heading points toward the top of the
        # image, then crop the forward-facing half (approximating the
        # CAMERA_FOV_DEG cone) and downscale to the final observation size.
        # pygame.transform.rotate is a visual CCW rotation: content at pixel
        # angle alpha (measured clockwise from +x, since py increases
        # downward — matching this env's world (dx,dy)=(cos,sin) heading
        # convention) moves to pixel angle (alpha - phi). "Ahead" starts at
        # pixel angle = heading_deg and must land at 270 deg (top, since
        # world-north/-y is "up" in this unrotated raster) => phi =
        # heading_deg - 270 = heading_deg + 90 (mod 360). Verified empirically
        # against dumped frames at heading 0/90/180/270 (see Phase 3 notes) —
        # do not "simplify" this back to 90-heading_deg, that formula only
        # agrees at heading 0/180 and points exactly backward at 90/270.
        rotated = pygame.transform.rotate(surface, math.degrees(self._robot_heading) + 90.0)
        rw, rh = rotated.get_size()
        ccx = rw // 2
        fov_half_width_px = int(win_px * math.tan(math.radians(CAMERA_FOV_DEG / 2)))
        crop_w = max(1, min(rw, 2 * fov_half_width_px, win_px))
        crop_h = max(1, win_px // 2)
        crop_rect = pygame.Rect(ccx - crop_w // 2, rh // 2 - crop_h, crop_w, crop_h)
        crop_rect = crop_rect.clip(pygame.Rect(0, 0, rw, rh))
        if crop_rect.width > 0 and crop_rect.height > 0:
            cropped = rotated.subsurface(crop_rect).copy()
        else:
            cropped = rotated

        scaled = pygame.transform.smoothscale(cropped, (CAMERA_IMG_SIZE, CAMERA_IMG_SIZE))
        return np.transpose(
            np.array(pygame.surfarray.array3d(scaled), dtype=np.uint8),
            axes=(1, 0, 2),
        )

    def _floor_color_grid(self) -> np.ndarray:
        """
        (height_cells, width_cells, 3) uint8 raster of the static floor/zone
        colors (walls/occupied cells dark grey, zones per ZONE_COLORS).

        FloorPlan.zone_map/occupancy are built once at construction and never
        mutated afterwards, so this is computed lazily once per env instance.
        The camera/rgb renders then sample it vectorized instead of issuing
        one pygame.draw.rect call per cell (~10k calls per camera frame — the
        dominant per-step cost behind the documented 171 -> 22 steps/s drop).
        """
        if self._floor_rgb is None:
            lut = np.empty((int(self.fp.zone_map.max()) + 1, 3), dtype=np.uint8)
            for code in range(lut.shape[0]):
                lut[code] = ZONE_COLORS.get(code, (220, 220, 220))
            rgb = lut[self.fp.zone_map].copy()
            rgb[self.fp.occupancy.astype(bool)] = (80, 80, 80)
            self._floor_rgb = rgb
        return self._floor_rgb

    def _world_to_px(self, x: float, y: float) -> Tuple[int, int]:
        px = int(x / self.fp.cell_size) * self._cell_px + self._cell_px // 2
        py = int(y / self.fp.cell_size) * self._cell_px + self._cell_px // 2
        return px, py

    def _draw_floor(self):
        import pygame
        for r in range(0, self.fp.height_cells, 2):
            for c in range(0, self.fp.width_cells, 2):
                code = int(self.fp.zone_map[r, c])
                color = (80,80,80) if self.fp.occupancy[r,c] else ZONE_COLORS.get(code, (220,220,220))
                rect = pygame.Rect(c*self._cell_px, r*self._cell_px, 2*self._cell_px, 2*self._cell_px)
                pygame.draw.rect(self._screen, color, rect)

    def _draw_pedestrians(self):
        import pygame
        for agent in self._crowd.agents:
            px, py = self._world_to_px(agent.pos[0], agent.pos[1])
            color = PCLASS_COLORS.get(agent.pclass, (100, 100, 100))
            r_px = max(3, int(agent.radius / self.fp.cell_size * self._cell_px))
            pygame.draw.circle(self._screen, color, (px, py), r_px)

    def _draw_goal(self):
        import pygame
        gx, gy = self._world_to_px(self._task_goal[0], self._task_goal[1])
        pygame.draw.circle(self._screen, (255, 200, 0), (gx, gy), 8)
        pygame.draw.circle(self._screen, (0, 0, 0), (gx, gy), 8, 2)

    def _draw_robot(self):
        import pygame
        rx, ry = self._world_to_px(self._robot_pos[0], self._robot_pos[1])
        r_px = max(4, int(ROBOT_RADIUS / self.fp.cell_size * self._cell_px))
        pygame.draw.circle(self._screen, (255, 60, 60), (rx, ry), r_px)
        hx = rx + int(r_px * 1.5 * math.cos(self._robot_heading))
        hy = ry + int(r_px * 1.5 * math.sin(self._robot_heading))
        pygame.draw.line(self._screen, (0, 0, 0), (rx, ry), (hx, hy), 2)

    def _draw_hud(self):
        import pygame
        W = self.fp.width_cells * self._cell_px
        H = self.fp.height_cells * self._cell_px
        pygame.draw.rect(self._screen, (30, 30, 30), pygame.Rect(0, H, W, 80))
        font = pygame.font.SysFont("monospace", 12)
        density = self._crowd.crowd_density(self._robot_pos)
        texts = [
            f"Task: {self._task_id}  Urgency:{self._urgency:.1f}",
            f"Step:{self._step}/{MAX_STEPS}  Zone:{self.fp.get_zone(*self._robot_pos)}",
            f"Density:{density:.2f}  ToD:{self._tod_hour:.1f}h",
            f"CultViol:{self._cultural_violations}  SocViol:{self._social_violations}",
        ]
        for i, t in enumerate(texts):
            surf = font.render(t, True, (220, 220, 220))
            self._screen.blit(surf, (6, H + 4 + i * 18))


# -------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------

def _make_env(**kwargs):
    return LMICHospitalNavEnv(**kwargs)


try:
    gym.register(
        id="LMICHospitalNav-v1",
        entry_point="envs.lmic_hospital_env:LMICHospitalNavEnv",
        max_episode_steps=MAX_STEPS,
    )
except Exception:
    pass  # already registered
