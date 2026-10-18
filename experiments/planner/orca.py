"""ORCA (Optimal Reciprocal Collision Avoidance, van den Berg et al. 2011) local
controller — the second classical baseline, built on a DIFFERENT principle than social
force (velocity obstacles + linear programming instead of potential forces).

Faithful condensed implementation for K tracked neighbours + LiDAR-derived static
points: each neighbour induces a half-plane constraint on the new velocity (the
standard truncated-VO geometry with reciprocity 1/2 for agents, 1 for static points);
the new velocity is the point of the feasible region closest to the preferred velocity
(toward the route planner's waypoint), found by incremental 2-D linear programming with
a sampling fallback when the LP is infeasible (dense-crowd case).

Non-holonomic adaptation (disclosed, identical discipline to the social-force baseline
AckermannSF): the chosen holonomic velocity v* is tracked with the pure-pursuit law
err -> omega = clip(2 err), speed = |v*| * max(vmin, 1 - |err|/pi), steering via the
bicycle-model inverse. Inputs are exactly the policy's observation: 5 nearest
pedestrians with velocities, the 48-ray LiDAR, own pose/speed, planner waypoint.
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple

import numpy as np

from baselines.classical import _omega_to_steer
from envs.lmic_hospital_env import (LIDAR_FOV, N_LIDAR_RAYS, LIDAR_MAX, PED_NEIGHBOR_RADIUS, PED_MAX_SPEED,
                                    ROBOT_RADIUS, lidar_ray_angles)

_ANGLES = lidar_ray_angles(LIDAR_FOV, N_LIDAR_RAYS)
PED_RADIUS = 0.25


def _cross2(a, b) -> float:
    """2-D cross product (numpy >= 2.0 removed np.cross for 2-vectors)."""
    return float(a[0] * b[1] - a[1] * b[0])


class OrcaController:
    def __init__(self, tau: float = 2.0, tau_static: float = 1.0, v_pref: float = 0.8,
                 vmin: float = 0.4, margin: float = 0.05, max_static: int = 12,
                 static_range: float = 3.0, max_omega: float = math.pi):
        self.tau, self.tau_s = tau, tau_static
        self.v_pref, self.vmin, self.margin = v_pref, vmin, margin
        self.max_static, self.static_range = max_static, static_range
        self.max_w = max_omega
        self._goal: Optional[np.ndarray] = None

    def set_goal(self, goal: np.ndarray) -> None:
        self._goal = np.asarray(goal, dtype=float).copy()

    # ---- ORCA geometry -------------------------------------------------
    @staticmethod
    def _halfplane(rel_p, rel_v, r, tau, dt=0.1):
        """Return (u, n): the smallest velocity change u (applied with the reciprocity
        share) and the outward normal n of the induced half-plane."""
        dist2 = rel_p @ rel_p
        r2 = r * r
        if dist2 > r2:                                    # no current overlap
            w = rel_v - rel_p / tau                       # relative to the cutoff-circle centre
            w2 = w @ w
            dot = w @ rel_p
            if dot < 0.0 and dot * dot > r2 * w2:         # project on the cutoff circle
                wl = math.sqrt(w2) or 1e-9
                n = w / wl
                u = (r / tau - wl) * n
            else:                                         # project on the nearer leg
                leg = math.sqrt(max(dist2 - r2, 1e-12))
                if _cross2(rel_p, w) > 0.0:               # left leg
                    d = np.array([rel_p[0] * leg - rel_p[1] * r, rel_p[0] * r + rel_p[1] * leg]) / dist2
                else:                                     # right leg
                    d = -np.array([rel_p[0] * leg + rel_p[1] * r, -rel_p[0] * r + rel_p[1] * leg]) / dist2
                u = (rel_v @ d) * d - rel_v
                n = np.array([-d[1], d[0]]) if _cross2(d, rel_v - (rel_v @ d) * d) < 0 else np.array([d[1], -d[0]])
                nl = np.linalg.norm(n) or 1e-9
                n = n / nl
                if n @ u < 0:
                    n = -n
        else:                                             # already overlapping: push out within one step
            w = rel_v - rel_p / dt
            wl = np.linalg.norm(w) or 1e-9
            n = w / wl
            u = (r / dt - wl) * n
        return u, n

    def _constraints(self, v_self, peds, statics):
        cons: List[Tuple[np.ndarray, np.ndarray]] = []    # (point, normal): feasible iff (v-point)@normal >= 0
        for rel_p, vel_j, r_j in peds:
            u, n = self._halfplane(rel_p, v_self - vel_j, ROBOT_RADIUS + r_j + self.margin, self.tau)
            cons.append((v_self + 0.5 * u, n))            # reciprocity 1/2: the other agent yields half
        for rel_p in statics:
            u, n = self._halfplane(rel_p, v_self.copy(), ROBOT_RADIUS + self.margin, self.tau_s)
            cons.append((v_self + u, n))                  # static points cannot yield
        return cons

    # ---- incremental 2-D LP (RVO2 linearProgram1/2, condensed) ---------
    def _lp(self, cons, v_pref, v_max):
        v = v_pref if np.linalg.norm(v_pref) <= v_max else v_pref * (v_max / np.linalg.norm(v_pref))
        for i, (pt, n) in enumerate(cons):
            if (v - pt) @ n >= -1e-9:
                continue
            # optimise along the line of constraint i: x = pt + t*dir, dir = rotate(n, 90)
            d = np.array([n[1], -n[0]])
            # intersection with the speed disc
            b = pt @ d
            c = pt @ pt - v_max * v_max
            disc = b * b - c
            if disc < 0:
                return None                               # line misses the disc -> infeasible
            sq = math.sqrt(disc)
            t_lo, t_hi = -b - sq, -b + sq
            for pt2, n2 in cons[:i]:
                den = d @ n2
                num = (pt2 - pt) @ n2
                if abs(den) < 1e-9:
                    if num > 0:
                        return None                       # parallel and infeasible
                    continue
                t = num / den
                if den > 0:
                    t_lo = max(t_lo, t)
                else:
                    t_hi = min(t_hi, t)
                if t_lo > t_hi:
                    return None
            t_star = float(np.clip((v_pref - pt) @ d, t_lo, t_hi))
            v = pt + t_star * d
        return v

    def _fallback(self, cons, v_pref, v_max):
        """Dense-crowd fallback: sampled velocity maximising the worst constraint margin."""
        best, best_key = np.zeros(2), (-1e18, 0.0)
        for sp in (v_max, 0.6 * v_max, 0.3 * v_max, 0.0):
            for k in range(16 if sp > 0 else 1):
                a = 2 * math.pi * k / 16
                cand = sp * np.array([math.cos(a), math.sin(a)])
                m = min(((cand - pt) @ n for pt, n in cons), default=0.0)
                key = (m, -np.linalg.norm(cand - v_pref))
                if key > best_key:
                    best, best_key = cand, key
        return best

    # ---- public API -----------------------------------------------------
    def act(self, robot_pos, robot_heading, robot_speed, ped_neighbors, lidar_hits):
        if self._goal is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        to_goal = self._goal - robot_pos
        dist = np.linalg.norm(to_goal)
        if dist < 0.05:
            return np.array([0.0, 0.0], dtype=np.float32)
        v_pref = to_goal / dist * min(self.v_pref, dist / 0.5)
        v_self = robot_speed * np.array([math.cos(robot_heading), math.sin(robot_heading)])

        peds = []
        for k in range(ped_neighbors.shape[0]):
            dx, dy = ped_neighbors[k, 0] * PED_NEIGHBOR_RADIUS, ped_neighbors[k, 1] * PED_NEIGHBOR_RADIUS
            if abs(dx) + abs(dy) < 1e-6:
                continue
            vel = np.array([ped_neighbors[k, 2], ped_neighbors[k, 3]]) * PED_MAX_SPEED
            peds.append((np.array([dx, dy]), vel, PED_RADIUS))

        statics = []
        hit = np.asarray(lidar_hits, float) * LIDAR_MAX
        sectors = {}
        for i, h in enumerate(hit):
            if h < self.static_range:
                s = i // 4                                  # 48 rays -> 12 sectors of 30 deg
                if s not in sectors or h < sectors[s][0]:
                    ang = robot_heading + _ANGLES[i]
                    sectors[s] = (h, np.array([h * math.cos(ang), h * math.sin(ang)]))
        statics = [p for _, p in sorted(sectors.values(), key=lambda t: t[0])[:self.max_static]]

        cons = self._constraints(v_self, peds, statics)
        v_new = self._lp(cons, v_pref, self.v_pref)
        if v_new is None:
            v_new = self._fallback(cons, v_pref, self.v_pref)

        speed = float(np.linalg.norm(v_new))
        if speed < 1e-3:
            return np.array([0.0, 0.0], dtype=np.float32)
        err = (math.atan2(v_new[1], v_new[0]) - robot_heading + math.pi) % (2 * math.pi) - math.pi
        omega = float(np.clip(2.0 * err, -self.max_w, self.max_w))
        v = min(speed, self.v_pref) * max(self.vmin, 1.0 - abs(err) / math.pi)
        return np.array([v, _omega_to_steer(omega, v)], dtype=np.float32)
