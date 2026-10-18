"""
Classical baselines for LMICHospitalNav-v1 comparison.

B1 — Static A* (no RL, pure path following)
B2 — Social Force navigation (HSFM-guided, no learning)
B3 — Vanilla SAC (no social/cultural reward terms)

All implement the same reward_fn signature:
  fn(obs, next_obs, action, info) -> float

AStarController/SocialForceController output Ackermann [v, steering_angle]
actions (matching envs.lmic_hospital_env's action space), converted from an
internal desired-yaw-rate planning quantity via _omega_to_steer().
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional

from envs.lmic_hospital_env import (
    LIDAR_FOV, N_LIDAR_RAYS, LIDAR_MAX, WHEELBASE, MAX_STEER,
    PED_NEIGHBOR_RADIUS, lidar_ray_angles,
)

_V_EPS = 0.02  # m/s floor to avoid the atan2 conversion blowing up as v -> 0


def _omega_to_steer(omega_desired: float, v: float,
                     wheelbase: float = WHEELBASE, max_steer: float = MAX_STEER) -> float:
    """
    Convert a desired yaw rate (as a differential-drive-style controller
    would compute) into an Ackermann/bicycle-model steering angle, via the
    inverse of heading_rate = (v/wheelbase)*tan(steering_angle).
    atan2 (not atan(.../.)) is used deliberately so this can never divide by
    zero even if the v_eff floor were bypassed, and it naturally preserves sign.
    """
    # The classical controllers only ever command v >= 0, but the env now
    # permits reverse (MIN_LINEAR < 0) — guard with abs() so a future caller
    # passing v < 0 gets a correctly-signed steering angle for backing up
    # (bicycle model: reversing flips the heading-rate sign via v, not delta).
    v_eff = max(abs(v), _V_EPS)
    delta = math.atan2(omega_desired * wheelbase, v_eff)
    return float(np.clip(delta, -max_steer, max_steer))


# ---------------------------------------------------------------------------
# B3 — Vanilla SAC reward (navigation + safety, no cultural)
# ---------------------------------------------------------------------------

class VanillaSACReward:
    """
    Navigation + collision penalty only (no social or cultural terms).
    B3 baseline in the ablation study.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        goal_bonus: float = 5.0,
        collision_penalty: float = -5.0,
    ) -> None:
        self.gamma = gamma
        self.goal_bonus = goal_bonus
        self.collision_penalty = collision_penalty
        self._prev_dist: Optional[float] = None

    def __call__(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        info: dict,
    ) -> float:
        if info["collision"]:
            self._prev_dist = None
            return self.collision_penalty

        # First step of an episode (env.reset() calls self.reset()) yields
        # zero progress rather than a stale cross-episode difference.
        curr_dist = info["dist_to_goal"]
        prev_dist = self._prev_dist if self._prev_dist is not None else curr_dist
        self._prev_dist = curr_dist

        # Undiscounted progress — the γ·Φ' – Φ form leaks +(1-γ)·d per step
        # for staying far from the goal (see rewards/components.py). Scaled
        # identically to every other agent's reward (PROGRESS_SCALE).
        from rewards.components import PROGRESS_SCALE
        r_nav = PROGRESS_SCALE * (prev_dist - curr_dist)
        if info["task_complete"]:
            r_nav += self.goal_bonus

        return float(np.clip(r_nav, -5.0, 6.0))

    def reset(self):
        self._prev_dist = None


# ---------------------------------------------------------------------------
# B1 — Static A* path-following controller
# ---------------------------------------------------------------------------

class AStarController:
    """
    Pure A*-based path follower. Computes an A* path at episode start and
    follows waypoints with a pure-pursuit controller.

    Not an RL agent — used as a lower-bound baseline.
    Acts via the gymnasium action interface for consistent evaluation.
    """

    def __init__(
        self,
        lookahead_dist: float = 0.8,
        max_linear: float = 0.8,
        max_omega_desired: float = math.pi,
    ) -> None:
        self.lookahead = lookahead_dist
        self.max_v = max_linear
        # Internal desired-yaw-rate planning bound — NOT a physical actuator
        # limit (the real actuator limit is MAX_STEER on the steering angle,
        # applied inside _omega_to_steer).
        self.max_w = max_omega_desired
        self._path = None
        self._wp_idx = 0

    def plan(self, fp, start: np.ndarray, goal: np.ndarray) -> None:
        """Compute A* path and reset waypoint index."""
        self._path = fp.astar(start, goal)
        self._wp_idx = 0

    def act(
        self,
        robot_pos: np.ndarray,
        robot_heading: float,
    ) -> np.ndarray:
        """
        Pure-pursuit controller: return Ackermann [v, steering_angle] action.
        """
        if self._path is None or len(self._path) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Advance waypoint
        while self._wp_idx < len(self._path) - 1:
            wp = self._path[self._wp_idx]
            if np.linalg.norm(wp - robot_pos) < 0.3:
                self._wp_idx += 1
            else:
                break

        target = self._path[min(self._wp_idx, len(self._path) - 1)]
        dx, dy = target[0] - robot_pos[0], target[1] - robot_pos[1]
        desired_heading = math.atan2(dy, dx)
        angle_err = desired_heading - robot_heading
        angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi

        # Proportional controller (desired yaw rate), converted to steering
        omega = float(np.clip(2.0 * angle_err, -self.max_w, self.max_w))
        v = self.max_v * max(0.0, 1.0 - abs(angle_err) / math.pi)
        steering_angle = _omega_to_steer(omega, v)

        return np.array([v, steering_angle], dtype=np.float32)

    @property
    def at_goal(self) -> bool:
        return self._wp_idx >= len(self._path or []) - 1


# ---------------------------------------------------------------------------
# B2 — Social Force controller (reactive obstacle avoidance, no learning)
# ---------------------------------------------------------------------------

class SocialForceController:
    """
    Reactive navigation using social forces for pedestrian avoidance.
    Combines A* goal direction with HSFM-style repulsion.

    This is a non-learning baseline that shows how classical methods
    compare to RL-based cultural adaptation.
    """

    def __init__(
        self,
        max_linear: float = 0.6,
        max_omega_desired: float = math.pi,
        A_rep: float = 2.0,
        B_rep: float = 0.3,
        goal_weight: float = 1.5,
        wall_weight: float = 3.0,
    ) -> None:
        self.max_v = max_linear
        # Internal desired-yaw-rate planning bound — NOT a physical actuator
        # limit (the real actuator limit is MAX_STEER, applied inside
        # _omega_to_steer).
        self.max_w = max_omega_desired
        self.A_rep = A_rep
        self.B_rep = B_rep
        self.goal_weight = goal_weight
        self.wall_weight = wall_weight
        self._goal: Optional[np.ndarray] = None

    def set_goal(self, goal: np.ndarray) -> None:
        self._goal = goal.copy()

    def act(
        self,
        robot_pos: np.ndarray,
        robot_heading: float,
        ped_neighbors: np.ndarray,  # (K, 4) [dx, dy, vx, vy] normalised — matches
                                    # obs["pedestrians"].reshape(K, 4)
        lidar_hits: np.ndarray,     # (N_LIDAR_RAYS,) normalised distances — matches
                                    # obs["lidar"]
    ) -> np.ndarray:
        """Return Ackermann [v, steering_angle] action."""
        if self._goal is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Goal force
        goal_dir = self._goal - robot_pos
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 0.01:
            goal_force = self.goal_weight * goal_dir / goal_dist
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Pedestrian repulsion — de-normalise dx,dy by the same
        # PED_NEIGHBOR_RADIUS used to normalise obs["pedestrians"]
        # (envs.lmic_hospital_env._ped_obs_vec), not a separate hardcoded value.
        rep_force = np.zeros(2)
        for k in range(ped_neighbors.shape[0]):
            dx = float(ped_neighbors[k, 0]) * PED_NEIGHBOR_RADIUS
            dy = float(ped_neighbors[k, 1]) * PED_NEIGHBOR_RADIUS
            dist = max(math.hypot(dx, dy), 0.1)
            rep_force += self.A_rep * np.exp(-dist / self.B_rep) * np.array([-dx, -dy]) / dist

        # Wall repulsion from LiDAR — ray angles/range imported from
        # envs.lmic_hospital_env so this can never silently diverge from the
        # env's own 360 deg / N_LIDAR_RAYS / LIDAR_MAX geometry.
        angles = lidar_ray_angles(LIDAR_FOV, N_LIDAR_RAYS)
        wall_force = np.zeros(2)
        for i, a in enumerate(angles):
            hit_dist = float(lidar_hits[i]) * LIDAR_MAX  # de-normalise
            if hit_dist < 1.5:
                wx = math.cos(robot_heading + a)
                wy = math.sin(robot_heading + a)
                mag = self.wall_weight * (1.5 - hit_dist) / 1.5
                wall_force += -mag * np.array([wx, wy])

        total_force = goal_force + rep_force + wall_force
        desired_heading = math.atan2(total_force[1], total_force[0])
        angle_err = desired_heading - robot_heading
        angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi

        omega = float(np.clip(2.0 * angle_err, -self.max_w, self.max_w))
        v = self.max_v * max(0.0, 1.0 - abs(angle_err) / (math.pi / 2))
        steering_angle = _omega_to_steer(omega, v)

        return np.array([v, steering_angle], dtype=np.float32)
