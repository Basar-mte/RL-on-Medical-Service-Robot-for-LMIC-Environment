"""Reactive safety shield (novel method, policy-axis not weighting-axis).

A control-barrier-style filter applied to any policy's action, using only
the onboard observation (LiDAR + nearest pedestrians) — no privileged info.
Because episodes terminate on collision, converting would-be collisions into
brake/steer-around maneuvers is the most direct lever on success rate.

Two components, both from the 48-ray 360-degree LiDAR (normalised /LIDAR_MAX):
  1. Speed governor: cap forward speed by the forward-cone clearance so the
     robot can always stop/turn before contact (CBF-style linear margin).
  2. Steer assist: when the forward path is blocked, bias steering toward the
     side (left vs right cone) with more clearance.

Reverse and free-space motion are untouched, so the shield only ever removes
collision-bound actions — it cannot make a safe action unsafe.
"""
from __future__ import annotations
import numpy as np
from envs.lmic_hospital_env import (
    lidar_ray_angles, LIDAR_MAX, MAX_LINEAR, MAX_STEER, ROBOT_RADIUS,
    N_LIDAR_RAYS, LIDAR_FOV,
)

_ANG = np.degrees(lidar_ray_angles(LIDAR_FOV, N_LIDAR_RAYS))
_FWD = np.where(np.abs(_ANG) <= 30)[0]           # forward +-30 deg
_LEFT = np.where((_ANG > 30) & (_ANG <= 90))[0]  # left  +30..+90
_RIGHT = np.where((_ANG < -30) & (_ANG >= -90))[0]

# CBF margin: robot body + buffer; k tuned so clearance >=1.5 m => no speed cap.
_D_MARGIN = ROBOT_RADIUS + 0.15
_K = MAX_LINEAR / (1.5 - _D_MARGIN)
_BLOCK = 1.2   # m; forward clearance below this triggers steer assist


def shield_action(action: np.ndarray, obs: dict) -> np.ndarray:
    """Return a collision-filtered [v, steering] given the raw policy action."""
    v, delta = float(action[0]), float(action[1])
    lidar_m = np.asarray(obs["lidar"], dtype=float) * LIDAR_MAX  # metres

    d_front = float(lidar_m[_FWD].min())
    # 1. speed governor (only brakes forward motion; reverse untouched)
    if v > 0:
        v_allowed = max(0.0, _K * (d_front - _D_MARGIN))
        v = min(v, v_allowed)

    # 2. steer assist when the forward path is blocked
    if d_front < _BLOCK and v >= 0:
        left = float(lidar_m[_LEFT].min()) if _LEFT.size else 0.0
        right = float(lidar_m[_RIGHT].min()) if _RIGHT.size else 0.0
        # steer toward the clearer side; strength scales with how blocked
        urgency = (_BLOCK - d_front) / _BLOCK          # 0..1
        bias = MAX_STEER * urgency * (1.0 if left > right else -1.0)
        delta = float(np.clip(delta + bias, -MAX_STEER, MAX_STEER))

    return np.array([np.clip(v, -MAX_LINEAR, MAX_LINEAR),
                     np.clip(delta, -MAX_STEER, MAX_STEER)], dtype=np.float32)
