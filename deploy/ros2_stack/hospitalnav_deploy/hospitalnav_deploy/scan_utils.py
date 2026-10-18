"""Shared LiDAR + shield utilities. Matches the sim exactly:
envs/lmic_hospital_env.py (ray geometry, constants) and
baselines/safety_shield.py (control-barrier filter)."""
from __future__ import annotations
import math
import numpy as np

# --- benchmark constants (must match envs/lmic_hospital_env.py) ---------------
N_RAYS      = 48
LIDAR_MAX   = 12.0                 # m
MAX_LINEAR  = 0.8                  # m/s
MIN_LINEAR  = -0.3                 # m/s
MAX_STEER   = math.radians(30)     # rad
ROBOT_RADIUS = 0.25                # m
DT          = 0.1                  # s -> 10 Hz control loop

# Ray bearings relative to robot heading; 0 = straight ahead, index 24 = forward.
BASE = np.linspace(-math.pi, math.pi, N_RAYS, endpoint=False)
_ANG = np.degrees(BASE)
_FWD   = np.where(np.abs(_ANG) <= 30)[0]            # forward +-30 deg
_LEFT  = np.where((_ANG > 30) & (_ANG <= 90))[0]
_RIGHT = np.where((_ANG < -30) & (_ANG >= -90))[0]
_D_MARGIN = ROBOT_RADIUS + 0.15                     # 0.40 m
_K = MAX_LINEAR / (1.5 - _D_MARGIN)                 # 0.727
_BLOCK = 1.2                                        # m


def scan_to_lidar48(ranges, angle_min, angle_increment, range_max=LIDAR_MAX,
                    bearing_offset=0.0, sector_min=False):
    """Downsample a LaserScan to the sim's 48-ray, [0,1]-normalized vector.

    bearing_offset (rad): add if the LiDAR's 0 deg is not the robot's +x axis.
    sector_min=False: the single beam nearest each of the 48 bearings — exactly the
        thin rays the policy was trained on (use for the POLICY input).
    sector_min=True: the minimum over every beam in each 7.5 deg sector — nothing
        between the sampled beams can be missed (use for the SAFETY SHIELD).
    """
    r = np.asarray(ranges, dtype=float)
    r[~np.isfinite(r)] = range_max
    r = np.minimum(r, range_max)
    n = len(r)
    out = np.empty(N_RAYS, dtype=np.float32)
    if sector_min:
        ang = angle_min + np.arange(n) * angle_increment - bearing_offset
        half = math.pi / N_RAYS                                   # half a sector
        for i, b in enumerate(BASE):
            d = (ang - b + math.pi) % (2 * math.pi) - math.pi
            sel = np.abs(d) <= half
            out[i] = r[sel].min() if sel.any() else range_max
    else:
        for i, b in enumerate(BASE):
            j = int(round(((b + bearing_offset) - angle_min) / angle_increment)) % n
            out[i] = r[j]
    return (out / LIDAR_MAX).astype(np.float32)


def shield(v: float, delta: float, lidar01: np.ndarray):
    """Control-barrier filter (mirrors baselines/safety_shield.py).
    Only removes collision-bound actions; never makes a safe action unsafe."""
    m = np.asarray(lidar01, dtype=float) * LIDAR_MAX
    d_front = float(m[_FWD].min())
    if v > 0:                                        # speed governor
        v = min(v, max(0.0, _K * (d_front - _D_MARGIN)))
    if d_front < _BLOCK and v >= 0:                  # steer assist
        left  = float(m[_LEFT].min())  if _LEFT.size  else 0.0
        right = float(m[_RIGHT].min()) if _RIGHT.size else 0.0
        urgency = (_BLOCK - d_front) / _BLOCK
        delta += MAX_STEER * urgency * (1.0 if left > right else -1.0)
    return (float(np.clip(v, -MAX_LINEAR, MAX_LINEAR)),
            float(np.clip(delta, -MAX_STEER, MAX_STEER)))


# --- v2: kinematic swept-path shield (mirrors experiments/planner/arc_shield.py) ---------
# Governs speed by the arc length to the first obstacle inside the body's swept band along
# the COMMANDED steering arc (bicycle model), over all LiDAR points and the tracked people
# propagated with their velocity; tries neighbouring steering angles when the commanded arc
# is blocked; reverse governed along the backward arc. Removed the forward cone's blind
# spot: zero collisions for every driver in the benchmark (Oct 2026).
WHEELBASE       = 0.20     # the TRAINING wheelbase (curvature semantics, see §3 of the deploy doc)
MIN_LINEAR_V2   = MIN_LINEAR
PED_R, PED_VMAX = 5.0, 2.5 # people_tracker normalisation (robot frame)
PED_RADIUS      = 0.25
MARGIN_LAT, MARGIN_LAT_PED, MARGIN_FWD = 0.03, 0.15, 0.15
HORIZON         = 0.6
STEER_ALTS      = (0.0, 0.15, -0.15, 0.3, -0.3)
_COS, _SIN      = np.cos(BASE), np.sin(BASE)
_SQ2            = math.sqrt(2.0)


def _points_v2(lidar01, people20):
    """Obstacle points in the ROBOT frame (x forward, y left). people20: people_tracker's
    [dx,dy,vx,vy]x5, already in the robot frame and normalised."""
    lidar = np.asarray(lidar01, dtype=float) * LIDAR_MAX
    hit = lidar < LIDAR_MAX - 1e-6
    pts = [np.stack([lidar[hit] * _COS[hit], lidar[hit] * _SIN[hit]], 1)]
    rad = [np.zeros(int(hit.sum()))]
    ped = np.asarray(people20, dtype=float).reshape(-1, 4)
    live = np.abs(ped[:, :2]).sum(1) > 1e-6
    if live.any():
        p, vel = ped[live, :2] * PED_R, ped[live, 2:] * PED_VMAX
        for t in (0.0, 0.5 * HORIZON, HORIZON):
            pts.append(p + vel * t); rad.append(np.full(len(p), PED_RADIUS))
    return np.concatenate(pts, 0), np.concatenate(rad, 0)


def _free_arc_v2(pts, rad, delta, forward=True):
    if len(pts) == 0:
        return math.inf
    x, y = pts[:, 0], pts[:, 1]
    if not forward:
        x = -x
    clear = ROBOT_RADIUS + rad + np.where(rad > 0, MARGIN_LAT_PED, MARGIN_LAT)
    if abs(delta) < 1e-3:
        inband = (np.abs(y) < clear) & (x > 0)
        if not inband.any():
            return math.inf
        return float(np.min(x[inband] - np.sqrt(np.maximum(clear[inband] ** 2 - y[inband] ** 2, 0.0))))
    rt = WHEELBASE / math.tan(delta)
    sgn = 1.0 if rt > 0 else -1.0
    rp = np.hypot(x, y - rt)
    lat = np.abs(rp - abs(rt))
    inband = lat < clear
    if not inband.any():
        return math.inf
    theta = np.mod(sgn * (np.arctan2(y[inband] - rt, x[inband]) + sgn * math.pi / 2), 2 * math.pi)
    half = np.sqrt(np.maximum(clear[inband] ** 2 - lat[inband] ** 2, 0.0))
    return float(np.min(abs(rt) * theta - half))


def shield_v2(v: float, delta: float, lidar01: np.ndarray, people20: np.ndarray):
    """Kinematic swept-path shield: returns the filtered (v, steering). Never increases |v|."""
    pts, rad = _points_v2(lidar01, people20)
    forward = v >= 0
    best_v, best_delta = 0.0, float(delta)
    for k, off in enumerate(STEER_ALTS):
        d = float(np.clip(delta + off, -MAX_STEER, MAX_STEER))
        s_free = _free_arc_v2(pts, rad, d, forward)
        v_allowed = (_K * max(0.0, s_free - MARGIN_FWD) if math.isfinite(s_free)
                     else (MAX_LINEAR if forward else abs(MIN_LINEAR_V2)))
        v_cand = min(abs(v), v_allowed)
        if k == 0:
            best_v, best_delta = v_cand, d
            if v_cand >= abs(v) - 1e-9:
                break
        elif v_cand > best_v + 0.05:
            best_v, best_delta = v_cand, d
    v_out = best_v if forward else -best_v
    return (float(np.clip(v_out, MIN_LINEAR_V2, MAX_LINEAR)), float(np.clip(best_delta, -MAX_STEER, MAX_STEER)))
