"""Kinematic (swept-path) safety shield — v2 of baselines/safety_shield.py.

The v1 shield governs speed by the clearance in a fixed +-30 deg forward cone. The
collision anatomy of the shielded hierarchical policies (Oct 2026) showed the residual
collisions happen at full speed with the obstacle at the robot's SIDE or behind
(79-81 % side, 16-19 % rear, 0-5 % front), while passing or turning in the bed-lined
corridor: invisible to a forward cone. This shield instead predicts the body's swept
path along the COMMANDED arc (bicycle model, steering delta) and governs speed by the
arc length to the first obstacle inside the swept band:

    v <= K * max(0, s_free),  s_free = s_p - (ROBOT_RADIUS + margin)

over all LiDAR hit points (48 rays, exact distances) and the 5 nearest tracked
pedestrians, each propagated with its observed velocity over the horizon. If the
commanded arc is blocked, a few neighbouring steering angles are tried and the one with
the largest admissible speed is used (DWA-style steer assist). Reverse is governed the
same way along the backward arc. Only the policy's observation is used (no privileged
information), and the shield never increases |v|.

Interface identical to baselines.safety_shield.shield_action(action, obs) with
obs = {"lidar": 48 normalised rays, "pedestrians": (20,) normalised [dx,dy,vx,vy]x5}.
"""
from __future__ import annotations
import math

import numpy as np

from envs.lmic_hospital_env import (lidar_ray_angles, LIDAR_MAX, LIDAR_FOV, N_LIDAR_RAYS, MAX_LINEAR,
                                    MIN_LINEAR, MAX_STEER, ROBOT_RADIUS, WHEELBASE, PED_NEIGHBOR_RADIUS,
                                    PED_MAX_SPEED, K_NEIGHBORS)

_ANG = lidar_ray_angles(LIDAR_FOV, N_LIDAR_RAYS)
_COS, _SIN = np.cos(_ANG), np.sin(_ANG)
# Two different margins (first version used 0.40 m for both and froze policies that drive
# 0.3 m from a wall or beside a walking pedestrian: 55 % timeouts):
MARGIN_LAT = 0.03                  # lateral, static obstacles: only points the body would touch count
MARGIN_LAT_PED = 0.15              # lateral, pedestrians: people move; never pass within 15 cm of them
MARGIN_FWD = 0.15                  # along-track stopping margin (v1's margin)
PED_RADIUS = 0.25                  # pedestrian body radius assumed
K = MAX_LINEAR / (1.5 - ROBOT_RADIUS - MARGIN_FWD)   # 0.727 — v1's gain: full speed at >= 1.5 m
HORIZON = 0.6                      # s of pedestrian motion propagated
STEER_ALTS = (0.0, 0.15, -0.15, 0.3, -0.3)       # rad offsets tried when the commanded arc is blocked


def _points(obs):
    """Obstacle points in the robot frame (x forward, y left): LiDAR hits (< max range)
    and pedestrian positions at t = 0, H/2, H. The pedestrian block is WORLD-frame
    relative offsets/velocities (envs.pedestrians.get_local_neighbors), so it is rotated
    by the heading carried in obs["robot_state"] = [x_n, y_n, sin(h), cos(h), v_n]."""
    lidar = np.asarray(obs["lidar"], dtype=float) * LIDAR_MAX
    hit = lidar < LIDAR_MAX - 1e-6
    pts = [np.stack([lidar[hit] * _COS[hit], lidar[hit] * _SIN[hit]], 1)]
    rad = [np.full(int(hit.sum()), 0.0)]
    ped = np.asarray(obs.get("pedestrians", np.zeros(4 * K_NEIGHBORS)), dtype=float).reshape(-1, 4)
    live = np.abs(ped[:, :2]).sum(1) > 1e-6
    if live.any():
        rs = np.asarray(obs.get("robot_state", [0, 0, 0.0, 1.0, 0]), dtype=float)
        sh, ch = float(rs[2]), float(rs[3])
        rot = np.array([[ch, sh], [-sh, ch]])              # world -> robot frame (rotate by -heading)
        p = (ped[live, :2] * PED_NEIGHBOR_RADIUS) @ rot.T
        v = (ped[live, 2:] * PED_MAX_SPEED) @ rot.T
        for t in (0.0, 0.5 * HORIZON, HORIZON):
            pts.append(p + v * t)
            rad.append(np.full(len(p), PED_RADIUS))
    return np.concatenate(pts, 0), np.concatenate(rad, 0)


def _free_arc(pts, rad, delta, forward=True):
    """Arc length the body can travel along the arc of steering `delta` before touching
    any point (inf if none inside the swept band)."""
    if len(pts) == 0:
        return math.inf
    x, y = pts[:, 0], pts[:, 1]
    if not forward:
        # Reversing with steering delta traces the SAME circle (centre (0, rt)) in the
        # opposite sense; mirroring x maps that onto the forward arc with the same delta.
        x = -x
    clear = ROBOT_RADIUS + rad + np.where(rad > 0, MARGIN_LAT_PED, MARGIN_LAT)
    if abs(delta) < 1e-3:         # straight line
        inband = (np.abs(y) < clear) & (x > 0)
        if not inband.any():
            return math.inf
        return float(np.min(x[inband] - np.sqrt(np.maximum(clear[inband] ** 2 - y[inband] ** 2, 0.0))))
    rt = WHEELBASE / math.tan(delta)          # signed turn radius (left > 0); centre at (0, rt)
    sgn = 1.0 if rt > 0 else -1.0
    rp = np.hypot(x, y - rt)                   # radial distance of each point from the centre
    lat = np.abs(rp - abs(rt))                 # lateral offset from the arc
    inband = lat < clear
    if not inband.any():
        return math.inf
    # angle travelled along the arc from the start pose (origin, heading +x): the start
    # point sits at angle -sgn*pi/2 from the centre; left turns advance counter-clockwise.
    theta = sgn * (np.arctan2(y[inband] - rt, x[inband]) + sgn * math.pi / 2)
    theta = np.mod(theta, 2 * math.pi)
    s = abs(rt) * theta
    # subtract the along-arc half-width of the body circle at that lateral offset
    half = np.sqrt(np.maximum(clear[inband] ** 2 - lat[inband] ** 2, 0.0))
    return float(np.min(s - half))


def shield_action(action, obs):
    """Return the shielded [v, steering]."""
    v, delta = float(action[0]), float(action[1])
    pts, rad = _points(obs)
    forward = v >= 0
    best_v, best_delta = 0.0, delta
    for k, off in enumerate(STEER_ALTS):
        d = float(np.clip(delta + off, -MAX_STEER, MAX_STEER))
        s_free = _free_arc(pts, rad, d, forward=forward)
        v_allowed = (K * max(0.0, s_free - MARGIN_FWD) if math.isfinite(s_free)
                     else abs(MAX_LINEAR if forward else MIN_LINEAR))
        v_cand = min(abs(v), v_allowed)
        if k == 0:
            best_v, best_delta = v_cand, d
            if v_cand >= abs(v) - 1e-9:      # commanded arc is admissible: keep the policy's steering
                break
        elif v_cand > best_v + 0.05:          # only deviate for a clearly better arc
            best_v, best_delta = v_cand, d
    v_out = best_v if forward else -best_v
    return np.array([np.clip(v_out, MIN_LINEAR, MAX_LINEAR), np.clip(best_delta, -MAX_STEER, MAX_STEER)],
                    dtype=np.float32)
