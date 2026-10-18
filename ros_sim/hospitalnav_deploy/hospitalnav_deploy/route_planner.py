"""Route planner library (numpy only, no ROS): the deployment twin of
experiments/planner/planner.py, so the robot's policies see exactly the goal
observation they were trained with.

Cost-to-go field by Dijkstra (8-connected, no corner cutting) on the robot-inflated
occupancy grid, per-cell multiplier
    m = 1 + ratio * c(zone) / 0.8 + 0.3 [within 0.2 m of an obstacle]
with c(prayer) = 2/3 * intensity (intensity 1.0 inside a prayer window, 0.3 outside),
c(male/female ward) = 0.1/3 — the CARS reward's per-step cultural penalties over the
0.8 progress reward of one full-speed step. ratio = 3 is the trained setting for the
cultural (CARS) agents, ratio = 0 the plain shortest path for vanilla / he2025.
From the field: the remaining route length (the policy's distance channel) and a
look-ahead waypoint `lookahead` m along the route, pulled back to line of sight.
"""
from __future__ import annotations
import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

SQ2 = math.sqrt(2.0)
STEP_PROGRESS_REWARD = 0.8          # PROGRESS_SCALE 10 * MAX_LINEAR 0.8 m/s * DT 0.1 s
PRAYER_PENALTY = 2.0 / 3.0
GENDER_PENALTY = 0.1 / 3.0
WALL_NEAR_M, WALL_NEAR_COST = 0.2, 0.3
DEFAULT_PRAYER_WINDOWS = [(5.0, 6.5), (12.0, 13.0), (15.0, 16.0), (17.5, 18.5), (19.0, 20.0)]
ZONE_PRAYER, ZONE_MALE, ZONE_FEMALE = 5, 6, 7      # indices in perception._ZONES


def in_prayer_window(hour: float, windows=DEFAULT_PRAYER_WINDOWS) -> bool:
    return any(a <= hour < b for a, b in windows)


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    H, W = mask.shape
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            if (dr == 0 and dc == 0) or dr * dr + dc * dc > n * n:
                continue
            src = mask[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
            out[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)] |= src
    return out


def downsample_blocked(blocked: np.ndarray, factor: int) -> np.ndarray:
    """Conservative coarsening: a coarse cell is blocked if ANY fine cell in it is."""
    if factor <= 1:
        return blocked
    H, W = blocked.shape
    Hc, Wc = -(-H // factor), -(-W // factor)
    pad = np.zeros((Hc * factor, Wc * factor), dtype=bool)
    pad[:H, :W] = blocked
    return pad.reshape(Hc, factor, Wc, factor).any(axis=(1, 3))


def cost_grid(blocked: np.ndarray, res: float, zone_idx: Optional[np.ndarray], ratio: float,
              in_window: bool) -> np.ndarray:
    m = np.ones(blocked.shape, dtype=np.float64)
    if ratio > 0 and zone_idx is not None:
        intensity = 1.0 if in_window else 0.3
        m[zone_idx == ZONE_PRAYER] += ratio * PRAYER_PENALTY * intensity / STEP_PROGRESS_REWARD
        m[(zone_idx == ZONE_MALE) | (zone_idx == ZONE_FEMALE)] += ratio * GENDER_PENALTY / STEP_PROGRESS_REWARD
    near = dilate(blocked, max(1, int(round(WALL_NEAR_M / res)))) & ~blocked
    m[near] += WALL_NEAR_COST
    m[blocked] = np.inf
    return m


def _dijkstra(cost: np.ndarray, gr: int, gc: int, cell: float):
    H, W = cost.shape
    N = H * W
    cost_l = cost.ravel().tolist()
    free = [math.isfinite(v) for v in cost_l]
    dist, length, parent = [math.inf] * N, [math.inf] * N, [-1] * N
    g = gr * W + gc
    dist[g] = length[g] = 0.0
    heap = [(0.0, g)]
    push, pop = heapq.heappush, heapq.heappop
    nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2))
    while heap:
        d, u = pop(heap)
        if d > dist[u]:
            continue
        r, c = divmod(u, W)
        cu = cost_l[u]
        for dr, dc, s in nbrs:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            v = nr * W + nc
            if not free[v] or (dr and dc and not (free[r * W + nc] and free[nr * W + c])):
                continue
            step = s * cell
            nd = d + step * 0.5 * (cu + cost_l[v])
            if nd < dist[v]:
                dist[v] = nd
                length[v] = length[u] + step
                parent[v] = u
                push(heap, (nd, v))
    return np.asarray(length).reshape(H, W), parent


class RoutePlanner:
    """Grid: row = (y - oy) / res, col = (x - ox) / res (nav_msgs/OccupancyGrid convention)."""

    def __init__(self, blocked: np.ndarray, res: float, origin_xy, zone_idx: Optional[np.ndarray] = None,
                 ratio: float = 3.0, in_window: bool = False, lookahead: float = 2.5, los: bool = True):
        self.blocked = blocked.astype(bool)
        self.res, (self.ox, self.oy) = float(res), (float(origin_xy[0]), float(origin_xy[1]))
        self.H, self.W = self.blocked.shape
        self.cost = cost_grid(self.blocked, self.res, zone_idx, ratio, in_window)
        self.free = np.isfinite(self.cost)
        self.lookahead, self.los = float(lookahead), bool(los)
        self.length: Optional[np.ndarray] = None
        self.parent: Optional[list] = None
        self.goal_rc: Optional[Tuple[int, int]] = None

    # -- grid helpers ----------------------------------------------------
    def cell(self, x: float, y: float) -> Tuple[int, int]:
        r = min(max(int((y - self.oy) / self.res), 0), self.H - 1)
        c = min(max(int((x - self.ox) / self.res), 0), self.W - 1)
        return r, c

    def xy(self, flat: int) -> Tuple[float, float]:
        r, c = divmod(flat, self.W)
        return (c + 0.5) * self.res + self.ox, (r + 0.5) * self.res + self.oy

    def nearest_free(self, r: int, c: int, max_radius: int = 30) -> Optional[Tuple[int, int]]:
        if self.free[r, c]:
            return r, c
        for rad in range(1, max_radius + 1):
            r0, r1 = max(0, r - rad), min(self.H, r + rad + 1)
            c0, c1 = max(0, c - rad), min(self.W, c + rad + 1)
            sub = self.free[r0:r1, c0:c1]
            if sub.any():
                rr, cc = np.nonzero(sub)
                k = np.argmin((rr + r0 - r) ** 2 + (cc + c0 - c) ** 2)
                return int(rr[k] + r0), int(cc[k] + c0)
        return None

    # -- planning ----------------------------------------------------------
    def set_goal(self, x: float, y: float) -> bool:
        rc = self.nearest_free(*self.cell(x, y))
        if rc is None:
            return False
        self.goal_rc = rc
        self.length, self.parent = _dijkstra(self.cost, rc[0], rc[1], self.res)
        return True

    def _start(self, x: float, y: float) -> Optional[int]:
        if self.length is None:
            return None
        rc = self.nearest_free(*self.cell(x, y), max_radius=10)
        if rc is None or not np.isfinite(self.length[rc]):
            return None
        return rc[0] * self.W + rc[1]

    def remaining(self, x: float, y: float) -> Optional[float]:
        u = self._start(x, y)
        return None if u is None else float(self.length[divmod(u, self.W)])

    def route_cells(self, x: float, y: float, max_len: Optional[float] = None) -> List[int]:
        u = self._start(x, y)
        if u is None:
            return []
        out, acc, cur = [u], 0.0, u
        while True:
            p = self.parent[cur]
            if p < 0:
                break
            pr, pc = divmod(p, self.W)
            cr, cc = divmod(cur, self.W)
            acc += self.res * (SQ2 if (pr != cr and pc != cc) else 1.0)
            if max_len is not None and acc > max_len + 1e-9:
                break
            out.append(p)
            cur = p
        return out

    def _visible(self, a: int, b: int) -> bool:
        ar, ac = divmod(a, self.W)
        br, bc = divmod(b, self.W)
        n = 2 * max(abs(br - ar), abs(bc - ac)) + 1
        rs = np.rint(np.linspace(ar, br, n)).astype(int)
        cs = np.rint(np.linspace(ac, bc, n)).astype(int)
        return bool(self.free[rs, cs].all())

    def waypoint(self, x: float, y: float) -> Optional[Tuple[float, float, float]]:
        """(wx, wy, remaining route length in m) or None if the goal is unreachable."""
        cells = self.route_cells(x, y, max_len=self.lookahead)
        if not cells:
            return None
        k = len(cells) - 1
        if self.los:
            while k > 1 and not self._visible(cells[0], cells[k]):
                k -= 1
        wx, wy = self.xy(cells[k])
        return wx, wy, float(self.length[divmod(cells[0], self.W)])

    def route_xy(self, x: float, y: float) -> List[Tuple[float, float]]:
        return [self.xy(c) for c in self.route_cells(x, y)]
