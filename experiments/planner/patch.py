"""In-process planner patch for LMICHospitalNavEnv (no main-code change).

Replaces the goal observation's bearing with the bearing to a route waypoint and,
optionally (PLANNER_PROGRESS=1, for training), makes the progress reward geodesic:
`info["dist_to_goal"]` becomes the remaining ROUTE length (every reward function --
CARS, vanilla, he2025 -- reads that key for its progress term), and the env's
`_prev_dist` is kept consistent before each step. The Euclidean distance is kept in
`info["euclid_dist_to_goal"]`. Nothing else in the env changes: collision, crowd,
cultural/social violation counting, task completion and the SPL reference path are
the unmodified main-code logic.

Environment variables (read by apply_from_env):
    PLANNER_MODE      off | geo | cult        (geo = shortest path; cult = reward-derived zone costs)
    PLANNER_RATIO     w_cult / w_nav for cult (default 1.0 = equal weights)
    PLANNER_L         look-ahead distance, metres (default 2.5)
    PLANNER_DIST      euclid | geo  -> the goal observation's distance channel (default euclid)
    PLANNER_PROGRESS  1 | 0        -> geodesic progress reward (default 1; irrelevant at evaluation)
    PLANNER_LOS       1 | 0        -> pull the waypoint back to line of sight (default 1)
"""
from __future__ import annotations

import math
import os

import numpy as np

import envs.lmic_hospital_env as em
from planner.planner import Planner


def apply_planner_patch(mode: str = "cult", ratio: float = 1.0, lookahead: float = 2.5,
                        dist: str = "euclid", progress: bool = True, los: bool = True) -> dict:
    E = em.LMICHospitalNavEnv
    if getattr(E, "_planner_patched", False):
        return E._planner_cfg
    if mode == "geo":
        ratio = 0.0
    cfg = dict(mode=mode, ratio=float(ratio), L=float(lookahead), dist=dist,
               progress=bool(progress), los=bool(los))
    orig_get_info, orig_step = E._get_info, E.step

    # PLANNER_MAP: plan on THIS floor plan instead of the env's own (stale-map /
    # unmapped-obstacle robustness: the world has obstacles the planner cannot know).
    stale = {}

    def _field(self):
        pl = self.__dict__.get("_route_planner")
        if pl is None or getattr(self, "_route_planner_for", None) is not self.fp:
            map_override = os.environ.get("PLANNER_MAP")
            if map_override:
                if map_override not in stale:
                    from envs.floorplan import load_floorplan
                    stale[map_override] = load_floorplan(map_override)
                fp_plan = stale[map_override]
            else:
                fp_plan = self.fp
            pl = self._route_planner = Planner(fp_plan, ratio=cfg["ratio"])
            self._route_planner_for = self.fp
        return pl.field(self._task_goal, self._tod_hour)

    def _goal_vec(self):
        f = _field(self)
        wp = f.waypoint(self._robot_pos, cfg["L"], cfg["los"])
        if wp is None:
            wp = self._task_goal
        d = wp - self._robot_pos
        if abs(d[0]) + abs(d[1]) < 1e-6:          # standing on the waypoint cell
            d = self._task_goal - self._robot_pos
        bearing = math.atan2(d[1], d[0]) - self._robot_heading
        dist_m = f.geodesic(self._robot_pos) if cfg["dist"] == "geo" else None
        if dist_m is None:
            dist_m = float(np.linalg.norm(self._task_goal - self._robot_pos))
        return np.array([math.sin(bearing), math.cos(bearing),
                         min(dist_m / self.GOAL_DIST_NORM, 1.0)], dtype=np.float32)

    E._goal_vec = _goal_vec

    if cfg["progress"]:
        def _get_info(self, collision=False, task_complete=False):
            info = orig_get_info(self, collision=collision, task_complete=task_complete)
            info["euclid_dist_to_goal"] = info["dist_to_goal"]
            g = _field(self).geodesic(self._robot_pos)
            if g is not None:
                info["dist_to_goal"] = g
            return info

        def step(self, action):
            g = _field(self).geodesic(self._robot_pos)
            if g is not None:
                self._prev_dist = g
            return orig_step(self, action)

        E._get_info = _get_info
        E.step = step

    E._planner_cfg = cfg
    E._planner_patched = True
    print(f"[planner] patch applied: {cfg}", flush=True)
    return cfg


def apply_from_env() -> bool:
    mode = os.environ.get("PLANNER_MODE", "off")
    if mode == "off":
        return False
    apply_planner_patch(mode=mode,
                        ratio=float(os.environ.get("PLANNER_RATIO", "1.0")),
                        lookahead=float(os.environ.get("PLANNER_L", "2.5")),
                        dist=os.environ.get("PLANNER_DIST", "euclid"),
                        progress=os.environ.get("PLANNER_PROGRESS", "1") == "1",
                        los=os.environ.get("PLANNER_LOS", "1") == "1")
    return True


def wrap_make_env(ts_module) -> None:
    """Re-apply the patch inside every env-factory thunk (SubprocVecEnv safety)."""
    orig = ts_module.make_env

    def _make_env(*a, **kw):
        factory = orig(*a, **kw)

        def _thunk():
            apply_from_env()
            return factory()
        return _thunk
    ts_module.make_env = _make_env
