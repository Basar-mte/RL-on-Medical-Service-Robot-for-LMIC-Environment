"""
LMICHospitalNav-v1 — Gymnasium RL environment.

Composes FloorPlan + PedestrianCrowd + differential-drive robot into a
fully-specified, check_env-compliant Gymnasium environment.

Observation (concatenated float32 vector)
------------------------------------------
lidar       (48,)  – LiDAR ray distances, clipped to [0, 10] m
robot_state  (5,)  – [x_n, y_n, sin(θ), cos(θ), v_n]
pedestrians (20,)  – 5 nearest pedestrians, each (dx, dy, vx, vy)
context     (12,)  – [crowd_density, task_urgency, zone_onehot×8, sin(tod), cos(tod)]
Total: 85 dims

Action  (2,)
------
[v, ω]  v ∈ [0, 0.8] m/s linear,  ω ∈ [−π, π] rad/s angular

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
MAX_ANGULAR   = math.pi  # rad/s
DT            = 0.1    # s
MAX_STEPS     = 1000
N_LIDAR_RAYS  = 48
LIDAR_MAX     = 10.0   # m
LIDAR_FOV     = 270.0  # degrees
K_NEIGHBORS   = 5
OBS_DIM       = N_LIDAR_RAYS + 5 + K_NEIGHBORS * 4 + 12  # 85


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
    ) -> None:
        super().__init__()

        if floorplan_path is None:
            floorplan_path = Path(__file__).parent.parent / "configs" / "dmch_hospital.yaml"
        self.floorplan_path = Path(floorplan_path)
        self.fp = load_floorplan(self.floorplan_path)
        self.render_mode = render_mode

        # Injected reward / weight source
        self._reward_fn = reward_fn
        self._weight_source = weight_source

        # Spaces
        obs_low  = np.full(OBS_DIM, -1.0, dtype=np.float32)
        obs_high = np.full(OBS_DIM,  1.0, dtype=np.float32)
        # LiDAR range [0, 1] when normalised
        obs_low[:N_LIDAR_RAYS]  = 0.0
        obs_high[:N_LIDAR_RAYS] = 1.0
        # Context density + urgency
        obs_low[N_LIDAR_RAYS + 5 : N_LIDAR_RAYS + 5 + 2] = 0.0
        obs_high[N_LIDAR_RAYS + 5 : N_LIDAR_RAYS + 5 + 2] = 1.0
        # zone onehot, sin/cos tod
        obs_low[N_LIDAR_RAYS + 5 + 2 : ] = -1.0
        obs_high[N_LIDAR_RAYS + 5 + 2 : ] = 1.0
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -MAX_ANGULAR], dtype=np.float32),
            high=np.array([MAX_LINEAR, MAX_ANGULAR], dtype=np.float32),
        )

        # Internal state
        self._rng = np.random.default_rng(seed)
        self._crowd = PedestrianCrowd(self.fp, max_pedestrians=80, rng=self._rng)
        self._reset_state()

        # Pygame renderer
        self._screen = None
        self._clock = None
        self._cell_px = 16  # pixels per cell at 0.1 m

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
        density_key = opts.get("density", "medium")

        self._setup_episode(task_id, density_key)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v, omega = float(action[0]), float(action[1])

        # Kinematic update (differential drive)
        self._robot_heading += omega * DT
        self._robot_heading = (self._robot_heading + math.pi) % (2 * math.pi) - math.pi
        dx = v * math.cos(self._robot_heading) * DT
        dy = v * math.sin(self._robot_heading) * DT
        new_x = self._robot_pos[0] + dx
        new_y = self._robot_pos[1] + dy

        # Collision check
        collision = False
        if not self.fp.is_walkable(new_x, new_y):
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

        # Check goal
        dist_to_goal = float(np.linalg.norm(self._robot_pos - self._task_goal))
        task_complete = bool(dist_to_goal < 0.6)

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

        # Robot spawn at nurse station area
        spawn_pts = self.fp.robot_spawn_points
        spawn = spawn_pts[self._rng.integers(len(spawn_pts))]
        self._robot_pos = np.array(spawn, dtype=float)
        self._robot_heading = self._rng.uniform(0, 2 * math.pi)
        self._robot_speed = 0.0

        # Task goal from floor plan config
        task_def = self.fp.tasks.get(task_id, {})
        target_room = task_def.get("target_room_id", "emergency_dept")
        try:
            goal_center = self.fp.room_center(target_room)
        except KeyError:
            goal_center = self.fp.random_walkable_point(self._rng)
        self._task_goal = self.fp.nearest_walkable(goal_center[0], goal_center[1])

        # Pedestrian crowd
        counts = DENSITY_LEVELS.get(density_key, DENSITY_LEVELS["medium"])
        self._crowd.reset(counts, rng=np.random.default_rng(int(self._rng.integers(1e6))))

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
        self._prev_obs = self._get_obs()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        lidar = self._cast_lidar()
        robot_state = self._robot_state_vec()
        ped_obs = self._crowd.get_local_neighbors(self._robot_pos, k=K_NEIGHBORS).flatten()
        ctx = self._context_vec()
        obs = np.concatenate([lidar, robot_state, ped_obs, ctx]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _cast_lidar(self) -> np.ndarray:
        """
        Vectorized LiDAR: all rays sampled simultaneously via numpy.
        ~20x faster than sequential Python loop.
        """
        step = self.fp.cell_size          # 0.1 m per sample
        n_samples = int(LIDAR_MAX / step)  # 100 samples per ray

        # Ray directions: (N_RAYS,)
        base_angles = np.linspace(
            -math.radians(LIDAR_FOV / 2),
             math.radians(LIDAR_FOV / 2),
             N_LIDAR_RAYS,
        )
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

        # Pedestrian hits: check each ped against each sample (vectorized)
        if self._crowd.agents:
            ped_pos = np.array([a.pos for a in self._crowd.agents])   # (P, 2)
            ped_rad = np.array([a.radius for a in self._crowd.agents]) # (P,)
            # dist² from each sample to each ped: (N_RAYS, n_samples, P)
            ddx = xs[:, :, None] - ped_pos[None, None, :, 0]
            ddy = ys[:, :, None] - ped_pos[None, None, :, 1]
            d2  = ddx ** 2 + ddy ** 2                                 # (N_RAYS, n_s, P)
            ped_hit = (d2 < (ped_rad ** 2)[None, None, :]).any(axis=2) # (N_RAYS, n_s)
            blocked = wall_hit | ped_hit
        else:
            blocked = wall_hit

        # First blocked sample per ray (argmax on bool: returns index of first True)
        first_hit_idx = np.argmax(blocked, axis=1)       # (N_RAYS,)
        any_hit       = blocked.any(axis=1)               # (N_RAYS,)
        hits = np.where(any_hit, dists[first_hit_idx], LIDAR_MAX).astype(np.float32)

        return hits / LIDAR_MAX

    def _robot_state_vec(self) -> np.ndarray:
        x_n = (self._robot_pos[0] / self.fp.width_m) * 2 - 1
        y_n = (self._robot_pos[1] / self.fp.height_m) * 2 - 1
        v_n = self._robot_speed / MAX_LINEAR
        return np.array([x_n, y_n, math.sin(self._robot_heading),
                         math.cos(self._robot_heading), v_n], dtype=np.float32)

    def _context_vec(self) -> np.ndarray:
        density = self._crowd.crowd_density(self._robot_pos)
        zone_oh = self.fp.zone_onehot(self._robot_pos[0], self._robot_pos[1])
        tod_sin = math.sin(2 * math.pi * self._tod_hour / 24)
        tod_cos = math.cos(2 * math.pi * self._tod_hour / 24)
        ctx = np.concatenate([
            [density, self._urgency],
            zone_oh,
            [tod_sin, tod_cos],
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

        if cv:
            self._cultural_violations += 1
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
        social_viol = min_ped < 0.5  # within personal space threshold
        if social_viol:
            self._social_violations += 1

        zone = self.fp.get_zone(self._robot_pos[0], self._robot_pos[1])
        cult_zone = self.fp.is_cultural_zone(self._robot_pos[0], self._robot_pos[1])

        # Path length tracking
        move = np.linalg.norm(self._robot_pos - self._prev_pos)
        self._total_path_len += move
        self._prev_pos = self._robot_pos.copy()

        return dict(
            collision=collision,
            task_complete=task_complete,
            steps=self._step,
            social_violation=social_viol,
            cultural_violation=cult_zone,
            current_zone=zone,
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
        self._density_key = "medium"
        self._prev_obs = np.zeros(OBS_DIM, dtype=np.float32)

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

        # Draw floor cells
        ZONE_COLORS = {
            0: (50,  50,  50),   # wall
            1: (220, 220, 220),  # walkable
            2: (200, 200, 180),  # corridor
            3: (180, 230, 180),  # nurse
            4: (180, 200, 230),  # pharmacy
            5: (230, 220, 180),  # ward
            6: (200, 180, 230),  # icu
            7: (230, 180, 180),  # emergency
            8: (150, 200, 150),  # prayer
            9: (180, 210, 230),  # male ward
            10:(230, 210, 190),  # female ward
        }
        for r in range(self.fp.height_cells):
            for c in range(self.fp.width_cells):
                code = int(self.fp.zone_map[r, c])
                if self.fp.occupancy[r, c]:
                    color = (80, 80, 80)
                else:
                    color = ZONE_COLORS.get(code, (220, 220, 220))
                rect = pygame.Rect(c * self._cell_px, r * self._cell_px,
                                   self._cell_px, self._cell_px)
                pygame.draw.rect(surface, color, rect)

        return np.transpose(
            np.array(pygame.surfarray.array3d(surface), dtype=np.uint8),
            axes=(1, 0, 2),
        )

    def _world_to_px(self, x: float, y: float) -> Tuple[int, int]:
        px = int(x / self.fp.cell_size) * self._cell_px + self._cell_px // 2
        py = int(y / self.fp.cell_size) * self._cell_px + self._cell_px // 2
        return px, py

    def _draw_floor(self):
        import pygame
        ZONE_COLORS = {
            0: (80,  80,  80), 1: (220, 220, 220), 2: (200, 200, 160),
            3: (160, 220, 160), 4: (160, 200, 220), 5: (220, 210, 160),
            6: (200, 160, 220), 7: (220, 160, 160), 8: (130, 190, 130),
            9: (160, 200, 220), 10:(220, 200, 180),
        }
        for r in range(0, self.fp.height_cells, 2):
            for c in range(0, self.fp.width_cells, 2):
                code = int(self.fp.zone_map[r, c])
                color = (80,80,80) if self.fp.occupancy[r,c] else ZONE_COLORS.get(code, (220,220,220))
                rect = pygame.Rect(c*self._cell_px, r*self._cell_px, 2*self._cell_px, 2*self._cell_px)
                pygame.draw.rect(self._screen, color, rect)

    def _draw_pedestrians(self):
        import pygame
        PCLASS_COLORS = {
            "medical_staff":    (0, 100, 200),
            "patient":          (200, 100, 0),
            "family_attendant": (180, 50, 180),
            "wheelchair":       (50, 180, 50),
        }
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
