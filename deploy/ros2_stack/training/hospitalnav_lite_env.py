"""Fast, pure-Python training environment mirroring the HospitalNav deployment
exactly: same 90-D observation layout, same physical action units, same
scan_utils.shield() applied every step (train shielded, deploy shielded), same
40x30 m floor plan as the Gazebo world (training/layout.py). No physics engine —
a numpy bicycle-model integrator — so it runs orders of magnitude faster than
Gazebo, which is the whole point: train here, deploy the checkpoint to Gazebo.
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                 "..", "hospitalnav_deploy", "hospitalnav_deploy"))
from scan_utils import (N_RAYS, LIDAR_MAX, MAX_LINEAR, MIN_LINEAR, MAX_STEER,
                         ROBOT_RADIUS, DT, shield)  # noqa: E402

import layout  # noqa: E402

WHEELBASE = 0.20
GOAL_TOL = 0.4
MAX_STEPS = 250
PED_R = 5.0
PED_VMAX = 2.5
K_PEDS = 5
N_MOVING_PEDS = 2
BASE_ANGLES = np.linspace(-math.pi, math.pi, N_RAYS, endpoint=False)


def _ray_rect_hit(ox, oy, dx, dy, rect):
    x0, y0, x1, y1 = rect
    tmin, tmax = 0.0, LIDAR_MAX
    for (o, d, lo, hi) in ((ox, dx, x0, x1), (oy, dy, y0, y1)):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin if tmin > 1e-6 else None


def _ray_circle_hit(ox, oy, dx, dy, cx, cy, r):
    fx, fy = ox - cx, oy - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    t = (-b - sq) / (2 * a)
    if t < 1e-6:
        t = (-b + sq) / (2 * a)
    return t if t > 1e-6 else None


class HospitalNavLiteEnv(gym.Env):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0.0, 1.0, (N_RAYS,), np.float32),
            "robot_state": spaces.Box(-1.0, 1.0, (5,), np.float32),
            "pedestrians": spaces.Box(-1.0, 1.0, (K_PEDS * 4,), np.float32),
            "context": spaces.Box(-1.0, 1.0, (15,), np.float32),
            "goal": spaces.Box(-1.0, 1.0, (3,), np.float32),
        })
        self.action_space = spaces.Box(
            np.array([MIN_LINEAR, -MAX_STEER], np.float32),
            np.array([MAX_LINEAR, MAX_STEER], np.float32))
        self.dynamic_people = []  # [x, y, vx, vy]
        self.urgency = 0.9
        self.tod_hour = 12.0

    # ---------------------------------------------------------- helpers ----
    def _free_point(self):
        while True:
            x = self.rng.uniform(1.0, layout.MAP_W - 1.0)
            y = self.rng.uniform(1.0, layout.MAP_H - 1.0)
            if self._point_free(x, y):
                return x, y

    def _point_free(self, x, y, margin=0.35):
        for (x0, y0, x1, y1) in layout.WALLS:
            if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
                return False
        for (x0, y0, x1, y1) in layout.FURNITURE:
            if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
                return False
        for (px, py, pr) in layout.PEOPLE:
            if math.hypot(x - px, y - py) < pr + margin:
                return False
        return True

    def _all_people(self):
        pts = [(px, py) for (px, py, _r) in layout.PEOPLE]
        pts += [(p[0], p[1]) for p in self.dynamic_people]
        return pts

    def _lidar(self):
        out = np.empty(N_RAYS, np.float32)
        obstacles = [(w, "wall") for w in layout.WALLS]
        obstacles += [(w, "wall") for w in layout.FURNITURE]
        obstacles += [((px, py, pr), "circle") for (px, py, pr) in layout.PEOPLE]
        obstacles += [((p[0], p[1], 0.18), "circle") for p in self.dynamic_people]
        for i, ang in enumerate(BASE_ANGLES):
            a = self.yaw + ang
            dx, dy = math.cos(a), math.sin(a)
            best = LIDAR_MAX
            for shape, kind in obstacles:
                if kind == "wall":
                    t = _ray_rect_hit(self.x, self.y, dx, dy, shape)
                else:
                    cx, cy, r = shape
                    t = _ray_circle_hit(self.x, self.y, dx, dy, cx, cy, r)
                if t is not None and t < best:
                    best = t
            out[i] = min(best, LIDAR_MAX) / LIDAR_MAX
        return out

    def _context(self):
        near = sum(1 for (px, py) in self._all_people()
                   if math.hypot(px - self.x, py - self.y) < 8.0)
        d5 = min(near / 30.0, 1.0)
        d2 = min(near / 8.0, 1.0)
        zone = layout.zone_at(self.x, self.y)
        zone_oh = np.zeros(9, np.float32)
        zone_oh[layout.zone_index(zone)] = 1.0
        cd = self._cult_dist()
        tod = 2 * math.pi * self.tod_hour / 24.0
        return np.concatenate([[d5, self.urgency], zone_oh,
                               [math.sin(tod), math.cos(tod)], [d2, cd]]).astype(np.float32)

    def _cult_dist(self):
        best = 1e9
        for (x0, y0, x1, y1, name, _doors) in layout.ROOMS:
            if name not in layout.CULTURAL:
                continue
            dx = max(x0 - self.x, 0, self.x - x1)
            dy = max(y0 - self.y, 0, self.y - y1)
            best = min(best, math.hypot(dx, dy))
        return min(best / 10.0, 1.0)

    def _pedestrians_obs(self):
        pts = self._all_people()
        rel = []
        for (px, py) in pts:
            dx, dy = px - self.x, py - self.y
            d = math.hypot(dx, dy)
            if d > PED_R:
                continue
            rx = dx * math.cos(-self.yaw) - dy * math.sin(-self.yaw)
            ry = dx * math.sin(-self.yaw) + dy * math.cos(-self.yaw)
            rel.append((d, rx, ry))
        rel.sort(key=lambda t: t[0])
        arr = np.zeros((K_PEDS, 4), np.float32)
        for i, (_d, rx, ry) in enumerate(rel[:K_PEDS]):
            arr[i] = [rx / PED_R, ry / PED_R, 0.0, 0.0]
        return np.clip(arr, -1, 1).flatten()

    def _goal_obs(self):
        gx, gy = self.goal
        bearing = math.atan2(gy - self.y, gx - self.x) - self.yaw
        dist = math.hypot(gx - self.x, gy - self.y)
        return np.array([math.sin(bearing), math.cos(bearing),
                         min(dist / 25.0, 1.0)], np.float32)

    def _obs(self):
        rs = np.array([self.x / layout.MAP_W * 2 - 1, self.y / layout.MAP_H * 2 - 1,
                       math.sin(self.yaw), math.cos(self.yaw),
                       self.vel / MAX_LINEAR], np.float32)
        return {
            "lidar": self._lidar(),
            "robot_state": rs,
            "pedestrians": self._pedestrians_obs(),
            "context": self._context(),
            "goal": self._goal_obs(),
        }

    # ------------------------------------------------------------- gym API --
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x, self.y = self._free_point()
        # Goal sampling mirrors deployment: path_to_waypoint.py only ever
        # feeds the policy a ~1.8 m lookahead point along Nav2's already-
        # planned path, never a distant point in another room — Nav2 (not
        # this reactive policy) is what finds doors. So sample a nearby goal
        # instead of an unconstrained full-map hop that could land behind a
        # wall this policy has no way to route around.
        while True:
            ang = self.rng.uniform(-math.pi, math.pi)
            dist = self.rng.uniform(3.0, 8.0)
            gx = self.x + dist * math.cos(ang)
            gy = self.y + dist * math.sin(ang)
            if (1.0 <= gx <= layout.MAP_W - 1.0 and 1.0 <= gy <= layout.MAP_H - 1.0
                    and self._point_free(gx, gy)):
                self.goal = (gx, gy)
                break
        self.yaw = self.rng.uniform(-math.pi, math.pi)
        self.vel = 0.0
        self.steps = 0
        self.dynamic_people = []
        for _ in range(N_MOVING_PEDS):
            px, py = self._free_point()
            ang = self.rng.uniform(-math.pi, math.pi)
            self.dynamic_people.append([px, py, 0.5 * math.cos(ang), 0.5 * math.sin(ang)])
        self.prev_dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        return self._obs(), {}

    def step(self, action):
        v_raw, steer_raw = float(action[0]), float(action[1])
        lidar = self._lidar()
        v, steer = shield(v_raw, steer_raw, lidar)

        self.yaw += (v / WHEELBASE) * math.tan(steer) * DT
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        self.x += v * math.cos(self.yaw) * DT
        self.y += v * math.sin(self.yaw) * DT
        self.vel = v
        self.steps += 1

        for p in self.dynamic_people:
            p[0] += p[2] * DT
            p[1] += p[3] * DT
            if not (1.0 <= p[0] <= layout.MAP_W - 1.0):
                p[2] *= -1
            if not (1.0 <= p[1] <= layout.MAP_H - 1.0):
                p[3] *= -1

        dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        post_lidar = self._lidar()
        clearance = float(np.min(post_lidar)) * LIDAR_MAX
        collided = clearance < ROBOT_RADIUS + 0.05
        reached = dist < GOAL_TOL

        reward = (self.prev_dist - dist) * 5.0 - 0.01
        self.prev_dist = dist
        terminated = False
        if collided:
            reward -= 30.0
            terminated = True
        if reached:
            reward += 50.0
            terminated = True
        truncated = self.steps >= MAX_STEPS

        return self._obs(), reward, terminated, truncated, {
            "collided": collided, "reached": reached}
