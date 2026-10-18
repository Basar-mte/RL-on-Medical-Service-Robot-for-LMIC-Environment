"""Isaac-independent core for the Tier-2 evaluation testbed.

Everything here runs without Isaac Sim installed, so it can be unit-tested on
any machine: the 90-D observation assembler, the zone lookup, and the Ackermann
joint-target conversion. `eval_isaac.py` is the thin glue that feeds this from
a live Isaac stage.

The observation layout is pinned to `envs.lmic_hospital_env`:

    lidar        48   normalized ranges, 360 deg, 12 m max
    robot_state   5   [x_n, y_n, sin(yaw), cos(yaw), v/MAX_LINEAR]
    pedestrians  20   5 nearest x [dx, dy, vx, vy]
    goal          3   [sin(bearing), cos(bearing), min(dist/25, 1)]
    context      14   [rho5, urgency, zone_onehot(8), sin(tod), cos(tod), rho2, cult_d]

Run the self-test:
    python deploy/isaac/isaac_core.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from envs.lmic_hospital_env import (  # noqa: E402
    LIDAR_MAX, MAX_LINEAR, MAX_STEER, K_NEIGHBORS, PED_MAX_SPEED,
    PED_NEIGHBOR_RADIUS, N_LIDAR_RAYS,
)

# Ackermann geometry. These come from the URDF the Isaac articulation is built
# from, NOT from controller/ackermann.py -- the two disagree by ~8 mm and the
# physics follows the URDF. See deploy/isaac/check_urdf.py.
URDF_WHEELBASE = 0.1371
URDF_TRACK = 0.1395
URDF_WHEEL_RADIUS = 0.0412

# The policy's own bicycle model, used to recover the curvature it intended.
SIM_WHEELBASE = 0.20

ZONE_INDEX = {"walkable": 0, "corridor": 1, "nurse_station": 2, "pharmacy": 3,
              "ward": 4, "icu": 4, "emergency": 4,
              "prayer": 5, "male_ward": 6, "female_ward": 7}
CULTURAL = {"prayer", "male_ward", "female_ward"}


class ZoneMap:
    """Zone lookup and cultural-distance, delegated to the training FloorPlan.

    An earlier version re-derived zones from the YAML polygons. It disagreed
    with FloorPlan on 14% of sampled points -- every one of them inside a wall,
    which FloorPlan rasterizes as its own zone and a polygon test cannot see.
    Harmless in practice, but re-implementing a lookup the environment already
    owns is precisely how a deployment path drifts away from the training path.
    So we call the real thing: agreement is now exact by construction.
    """

    def __init__(self, cfg_path: Path):
        from envs.floorplan import load_floorplan
        self.fp = load_floorplan(str(cfg_path))
        self.w = float(self.fp.width_m)
        self.h = float(self.fp.height_m)

    def zone_at(self, x: float, y: float) -> str:
        return self.fp.get_zone(x, y)

    def onehot(self, x: float, y: float) -> np.ndarray:
        return self.fp.zone_onehot(x, y)

    def cultural_distance(self, x: float, y: float) -> float:
        """Metres to the nearest culturally sensitive zone, 0 inside one."""
        return float(self.fp.cultural_zone_distance(x, y))


def build_obs(pos, yaw, speed, lidar_ranges, people, goal, zones: ZoneMap,
              urgency: float, tod_hour: float) -> dict:
    """Assemble the 90-D observation dict the trained policies expect.

    pos          (2,)   robot position, metres, map frame
    yaw          float  heading, radians
    speed        float  forward speed, m/s
    lidar_ranges (48,)  metres; inf/nan treated as max range
    people       (N,4)  [x, y, vx, vy] in the map frame; may be empty
    goal         (2,)   goal position, metres, map frame
    """
    pos = np.asarray(pos, dtype=float)
    goal = np.asarray(goal, dtype=float)

    scan = np.asarray(lidar_ranges, dtype=np.float32).copy()
    scan[~np.isfinite(scan)] = LIDAR_MAX
    lidar = np.clip(scan, 0.0, LIDAR_MAX) / LIDAR_MAX

    robot_state = np.array([
        pos[0] / zones.w * 2 - 1,
        pos[1] / zones.h * 2 - 1,
        math.sin(yaw), math.cos(yaw),
        speed / MAX_LINEAR,
    ], dtype=np.float32)

    ped = np.zeros(K_NEIGHBORS * 4, dtype=np.float32)
    people = np.asarray(people, dtype=float).reshape(-1, 4)
    if len(people):
        d = np.linalg.norm(people[:, :2] - pos, axis=1)
        keep = np.argsort(d)[:K_NEIGHBORS]
        for i, j in enumerate(keep):
            if d[j] > PED_NEIGHBOR_RADIUS:
                break
            rel = people[j, :2] - pos
            ped[i * 4:i * 4 + 4] = [rel[0] / PED_NEIGHBOR_RADIUS,
                                    rel[1] / PED_NEIGHBOR_RADIUS,
                                    people[j, 2] / PED_MAX_SPEED,
                                    people[j, 3] / PED_MAX_SPEED]
        n5 = int((d <= 5.0).sum())
        n2 = int((d <= 2.0).sum())
    else:
        n5 = n2 = 0

    to_goal = goal - pos
    bearing = math.atan2(to_goal[1], to_goal[0]) - yaw
    goal_vec = np.array([math.sin(bearing), math.cos(bearing),
                         min(float(np.linalg.norm(to_goal)) / 25.0, 1.0)],
                        dtype=np.float32)

    tod = 2 * math.pi * tod_hour / 24.0
    context = np.concatenate([
        [min(n5 / 30.0, 1.0), urgency],
        zones.onehot(pos[0], pos[1]),
        [math.sin(tod), math.cos(tod)],
        [min(n2 / 8.0, 1.0), zones.cultural_distance(pos[0], pos[1]) / 10.0],
    ]).astype(np.float32)

    return {"lidar": lidar, "robot_state": robot_state, "pedestrians": ped,
            "goal": goal_vec, "context": context}


def ackermann_joint_targets(v: float, delta: float):
    """Policy action [v, steering angle] -> per-joint targets for the URDF.

    The policy's delta is defined on ITS wheelbase (0.20 m), so first recover
    the curvature it meant, then re-derive the angle on the real wheelbase.
    Left and right steer differently on a real linkage (Ackermann condition);
    the single-track bicycle model the policy uses has no notion of that, so we
    reconstruct both from the shared curvature.

    Returns (steer_left, steer_right, wheel_omega) in rad and rad/s.
    """
    v = float(np.clip(v, -MAX_LINEAR, MAX_LINEAR))
    delta = float(np.clip(delta, -MAX_STEER, MAX_STEER))

    kappa = math.tan(delta) / SIM_WHEELBASE          # 1/m, what the policy meant
    omega = v / URDF_WHEEL_RADIUS                    # rad/s of the driven wheels

    if abs(kappa) < 1e-9:
        return 0.0, 0.0, omega

    R = 1.0 / kappa                                  # signed turn radius at centre
    inner = math.atan(URDF_WHEELBASE / (R - math.copysign(URDF_TRACK / 2, R)))
    outer = math.atan(URDF_WHEELBASE / (R + math.copysign(URDF_TRACK / 2, R)))
    lo, hi = -MAX_STEER_URDF, MAX_STEER_URDF
    left, right = (inner, outer) if kappa > 0 else (outer, inner)
    return float(np.clip(left, lo, hi)), float(np.clip(right, lo, hi)), omega


MAX_STEER_URDF = math.radians(45.0)   # the URDF's revolute limit


# ---------------------------------------------------------------------------
# self-test: the assembler must agree with the training environment
# ---------------------------------------------------------------------------
def _self_test() -> int:
    import gymnasium  # noqa: F401
    from envs.lmic_hospital_env import LMICHospitalNavEnv

    cfg = ROOT / "configs" / "hospital_v4.yaml"
    env = LMICHospitalNavEnv(floorplan_path=str(cfg), use_camera=False)
    obs, _ = env.reset(seed=0)
    zones = ZoneMap(cfg)

    print("shapes vs. the training env:")
    ok = True
    # Feed the env's actual crowd: the density terms in the context vector are
    # counts within 5 m and 2 m, so an empty crowd would compare unequal.
    crowd = np.array([[a.pos[0], a.pos[1], 0.0, 0.0]
                      for a in env._crowd.agents], dtype=float)
    if crowd.size == 0:
        crowd = np.zeros((0, 4))
    ours = build_obs(pos=env._robot_pos, yaw=env._robot_heading, speed=0.0,
                     lidar_ranges=np.full(N_LIDAR_RAYS, LIDAR_MAX),
                     people=crowd, goal=env._task_goal,
                     zones=zones, urgency=env._urgency, tod_hour=env._tod_hour)
    for k in ("lidar", "robot_state", "pedestrians", "goal", "context"):
        a, b = np.shape(obs[k]), np.shape(ours[k])
        same = a == b
        ok &= same
        print(f"  {k:<12} env {a}  ours {b}   {'OK' if same else '<<< MISMATCH'}")
    total = sum(int(np.size(ours[k])) for k in ours)
    print(f"  total {total} dims  {'OK' if total == 90 else '<<< expected 90'}")
    ok &= total == 90

    print("\nagreement on the same state (goal + robot_state + context):")
    for k in ("robot_state", "goal", "context"):
        d = float(np.max(np.abs(np.asarray(obs[k]) - ours[k])))
        good = d < 1e-4
        ok &= good
        print(f"  {k:<12} max|diff| = {d:.2e}   {'OK' if good else '<<< DIVERGES'}")

    print("\nzone lookup vs. FloorPlan:")
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [zones.w, zones.h], size=(400, 2))
    agree = sum(np.argmax(zones.onehot(*p)) == np.argmax(env.fp.zone_onehot(*p))
                for p in pts)
    print(f"  {agree}/400 sample points agree "
          f"({'OK' if agree == 400 else '<<< must be exact now'})")
    ok &= agree == 400

    print("\nAckermann conversion:")
    for v, deg in ((0.8, 0.0), (0.8, 30.0), (0.8, -30.0), (-0.3, 20.0)):
        l, r, w = ackermann_joint_targets(v, math.radians(deg))
        kappa = math.tan(math.radians(deg)) / SIM_WHEELBASE
        print(f"  v={v:+.1f} delta={deg:+5.1f}deg -> steer L{math.degrees(l):+6.2f} "
              f"R{math.degrees(r):+6.2f} deg, wheel {w:+6.2f} rad/s "
              f"(kappa={kappa:+.3f} 1/m)")
        ok &= abs(l) <= MAX_STEER_URDF + 1e-9 and abs(r) <= MAX_STEER_URDF + 1e-9

    print("\nVERDICT:", "core matches the training environment" if ok
          else "MISMATCH -- do not evaluate until resolved")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
