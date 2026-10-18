"""Ackermann-adapted social-force local driver (robot-frame; deployment twin of
experiments/planner/sf_prior.AckermannSF, the benchmark's strongest classical
controller and the PRIOR of the residual RL agent).

Forces (all in the ROBOT frame, heading = +x):
  goal pull toward the route planner's waypoint, exponential repulsion from the
  5 tracked people, wall repulsion from LiDAR returns within 1.5 m.
Speed law: pure pursuit with a minimum rolling fraction vmin (an Ackermann chassis
cannot turn at v = 0 — the stock social-force speed law deadlocks at doorways).
Tuned gains (selected on held-out episodes, Oct 2026): goal 1.5, wall 1.0, vmin 0.4.
"""
from __future__ import annotations
import math

import numpy as np

from .scan_utils import BASE, LIDAR_MAX, MAX_LINEAR, _K  # noqa: F401  (shared constants)
from .scan_utils import MAX_STEER, WHEELBASE

PED_R = 5.0          # people_tracker normalisation (m)
PED_VMAX = 2.5       # (m/s)
TUNED = dict(goal_weight=1.5, wall_weight=1.0, vmin=0.4, v_max=0.8, A_rep=2.0, B_rep=0.3, max_omega=math.pi)


def _omega_to_steer(omega: float, v: float) -> float:
    v_eff = max(abs(v), 0.02)
    return float(np.clip(math.atan2(omega * WHEELBASE, v_eff), -MAX_STEER, MAX_STEER))


class AckermannSF:
    def __init__(self, goal_weight=1.5, wall_weight=1.0, vmin=0.4, v_max=0.8,
                 A_rep=2.0, B_rep=0.3, max_omega=math.pi):
        self.gw, self.ww, self.vmin, self.v_max = goal_weight, wall_weight, vmin, v_max
        self.A, self.B, self.max_w = A_rep, B_rep, max_omega

    @classmethod
    def tuned(cls):
        return cls(**TUNED)

    def act(self, goal_bearing: float, goal_dist: float, people20: np.ndarray, lidar01: np.ndarray):
        """All inputs in the ROBOT frame. people20 = people_tracker's [dx,dy,vx,vy]x5
        (normalised). Returns [v, steering]."""
        if goal_dist <= 0.05:
            return np.array([0.0, 0.0], dtype=np.float32)
        f = self.gw * np.array([math.cos(goal_bearing), math.sin(goal_bearing)])
        ped = np.asarray(people20, dtype=float).reshape(-1, 4)
        for k in range(ped.shape[0]):
            dx, dy = ped[k, 0] * PED_R, ped[k, 1] * PED_R
            if abs(dx) + abs(dy) < 1e-6:
                continue
            d = max(math.hypot(dx, dy), 0.1)
            f = f + self.A * math.exp(-d / self.B) * np.array([-dx, -dy]) / d
        hit = np.asarray(lidar01, dtype=float) * LIDAR_MAX
        near = hit < 1.5
        if near.any():
            mag = self.ww * (1.5 - hit[near]) / 1.5
            f = f - np.array([np.sum(mag * np.cos(BASE[near])), np.sum(mag * np.sin(BASE[near]))])
        err = (math.atan2(f[1], f[0]) + math.pi) % (2 * math.pi) - math.pi
        omega = float(np.clip(2.0 * err, -self.max_w, self.max_w))
        v = self.v_max * max(self.vmin, 1.0 - abs(err) / math.pi)
        return np.array([v, _omega_to_steer(omega, v)], dtype=np.float32)
