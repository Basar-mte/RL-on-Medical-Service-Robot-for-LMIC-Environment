"""Ackermann-aware social-force controller: the tuned classical local controller of the
final campaign (scripts/eval_classical_v5.py, `planner_sf2`) and the PRIOR of the residual
RL agent (experiments/planner/residual.py).

Same forces as baselines.classical.SocialForceController (goal pull toward the planner's
waypoint, exponential pedestrian repulsion, wall repulsion from LiDAR within 1.5 m), but a
pure-pursuit speed law with a minimum rolling speed: the stock speed law reaches v = 0
whenever the resultant force points > 90 deg off the heading, and an Ackermann chassis
cannot turn at v = 0 (deadlock; 6 % success). Gains selected on tuning episodes (seeds
5000+, Oct 2026): goal_weight 1.5, wall_weight 1.0, vmin 0.4.
"""
from __future__ import annotations
import math

import numpy as np

from baselines.classical import SocialForceController, _omega_to_steer
from envs.lmic_hospital_env import LIDAR_FOV, N_LIDAR_RAYS, LIDAR_MAX, PED_NEIGHBOR_RADIUS, lidar_ray_angles

TUNED = dict(goal_weight=1.5, wall_weight=1.0, vmin=0.4, max_linear=0.8)
_ANGLES = lidar_ray_angles(LIDAR_FOV, N_LIDAR_RAYS)


class AckermannSF(SocialForceController):
    def __init__(self, vmin=0.4, **kw):
        super().__init__(**kw)
        self.vmin = vmin

    @classmethod
    def tuned(cls):
        g = dict(TUNED)
        return cls(vmin=g.pop("vmin"), **g)

    def act(self, robot_pos, robot_heading, ped_neighbors, lidar_hits):
        if self._goal is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        goal_dir = self._goal - robot_pos
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist <= 0.01:
            return np.array([0.0, 0.0], dtype=np.float32)
        force = self.goal_weight * goal_dir / goal_dist
        for k in range(ped_neighbors.shape[0]):
            dx = float(ped_neighbors[k, 0]) * PED_NEIGHBOR_RADIUS
            dy = float(ped_neighbors[k, 1]) * PED_NEIGHBOR_RADIUS
            dist = max(math.hypot(dx, dy), 0.1)
            force = force + self.A_rep * np.exp(-dist / self.B_rep) * np.array([-dx, -dy]) / dist
        hit = np.asarray(lidar_hits, float) * LIDAR_MAX
        near = hit < 1.5
        if near.any():
            mag = self.wall_weight * (1.5 - hit[near]) / 1.5
            force = force - np.array([np.sum(mag * np.cos(robot_heading + _ANGLES[near])),
                                      np.sum(mag * np.sin(robot_heading + _ANGLES[near]))])
        err = (math.atan2(force[1], force[0]) - robot_heading + math.pi) % (2 * math.pi) - math.pi
        omega = float(np.clip(2.0 * err, -self.max_w, self.max_w))
        v = self.max_v * max(self.vmin, 1.0 - abs(err) / math.pi)
        return np.array([v, _omega_to_steer(omega, v)], dtype=np.float32)
