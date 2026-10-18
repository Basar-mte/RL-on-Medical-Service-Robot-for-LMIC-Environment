"""Global route planner for the LMIC-HospitalNav maps (experiment-only; no main-code change).

A cost-to-go field is computed by Dijkstra (8-connected, no corner cutting) on the
ROBOT-inflated occupancy grid of the floor plan, from the task goal outwards, with a
per-cell cost multiplier

    m(x) = 1 + ratio * c(x) / (PROGRESS_SCALE * MAX_LINEAR * DT) + wall-proximity term

where c(x) is the CARS per-step cultural penalty the policy's reward charges at x
(prayer room 2.0*intensity/3, gender wards 0.1/3 -- the 1/3 is cultural_reward's
default term weighting) and PROGRESS_SCALE*MAX_LINEAR*DT = 0.8 is the progress reward
one full-speed step earns. With ratio = w_cult / w_nav (1 for equal weights) the
planner minimises exactly the scalarised objective the reward integrates along a
route, so the hierarchical agent optimises ONE objective: the route by the planner,
the local motion (crowd, doors, timing) by RL. ratio = 0 is the plain shortest path
(geometric planner, for culture-blind agents).

The field gives, for any robot position: the remaining route length ("geodesic"
distance, metres) and a look-ahead waypoint `lookahead` metres along the route
(pure-pursuit style, optionally pulled back to the last point with line of sight).
Fields are cached per (floor plan, ratio, goal cell, prayer-window state).
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from envs.floorplan import FloorPlan, ZONE_CODES
from envs.lmic_hospital_env import MAX_LINEAR, DT
from rewards.components import PROGRESS_SCALE

SQ2 = math.sqrt(2.0)
STEP_PROGRESS_REWARD = PROGRESS_SCALE * MAX_LINEAR * DT   # 0.8 reward per full-speed step
PRAYER_PENALTY = 2.0 / 3.0   # per step inside the prayer room, x intensity (1.0 in window, 0.3 outside)
GENDER_PENALTY = 0.1 / 3.0   # per step inside a gender-segregated ward
WALL_NEAR_CELLS, WALL_NEAR_COST = 2, 0.3   # +0.3 within 0.2 m of the inflated obstacle boundary


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    H, W = mask.shape
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            if dr == 0 and dc == 0:
                continue
            src = mask[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
            out[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)] |= src
    return out


def cost_grid(fp: FloorPlan, ratio: float, in_prayer_window: bool) -> np.ndarray:
    """Per-cell cost multiplier (inf = not traversable by the robot centre)."""
    occ = fp.occupancy_inflated.astype(bool)
    m = np.ones(occ.shape, dtype=np.float64)
    if ratio > 0:
        intensity = 1.0 if in_prayer_window else 0.3
        prayer = fp.zone_map == ZONE_CODES["prayer"]
        gender = (fp.zone_map == ZONE_CODES["male_ward"]) | (fp.zone_map == ZONE_CODES["female_ward"])
        m[prayer] += ratio * PRAYER_PENALTY * intensity / STEP_PROGRESS_REWARD
        m[gender] += ratio * GENDER_PENALTY / STEP_PROGRESS_REWARD
    near = _dilate(occ, WALL_NEAR_CELLS) & ~occ
    m[near] += WALL_NEAR_COST
    m[occ] = np.inf
    return m


def _dijkstra(cost: np.ndarray, gr: int, gc: int, cell: float):
    """Cost-to-go, metric route length and parent pointer for every cell (flat index)."""
    H, W = cost.shape
    N = H * W
    cost_l = cost.ravel().tolist()
    free = [math.isfinite(v) for v in cost_l]
    dist = [math.inf] * N
    length = [math.inf] * N
    parent = [-1] * N
    g = gr * W + gc
    dist[g] = 0.0
    length[g] = 0.0
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
            if not free[v]:
                continue
            if dr and dc and not (free[r * W + nc] and free[nr * W + c]):
                continue   # no corner cutting past an obstacle cell
            step = s * cell
            nd = d + step * 0.5 * (cu + cost_l[v])
            if nd < dist[v]:
                dist[v] = nd
                length[v] = length[u] + step
                parent[v] = u
                push(heap, (nd, v))
    return (np.asarray(dist).reshape(H, W), np.asarray(length).reshape(H, W),
            np.asarray(parent, dtype=np.int64))


class CostField:
    def __init__(self, fp: FloorPlan, cost: np.ndarray, goal_xy) -> None:
        self.fp, self.cell = fp, fp.cell_size
        self.H, self.W = cost.shape
        self.free = np.isfinite(cost)
        gr, gc = fp._world_to_cell(goal_xy[0], goal_xy[1])
        if not self.free[gr, gc]:
            nw = fp.nearest_walkable(goal_xy[0], goal_xy[1], grid=fp.occupancy_inflated)
            gr, gc = fp._world_to_cell(nw[0], nw[1])
        self.goal_rc = (gr, gc)
        self.cost_to_go, self.length, self.parent = _dijkstra(cost, gr, gc, self.cell)
        self._parent_l = self.parent.tolist()

    # -- lookups --------------------------------------------------------
    def _cell(self, xy) -> Optional[Tuple[int, int]]:
        r, c = self.fp._world_to_cell(xy[0], xy[1])
        if not np.isfinite(self.length[r, c]):
            nw = self.fp.nearest_walkable(xy[0], xy[1], max_radius=10, grid=self.fp.occupancy_inflated)
            r, c = self.fp._world_to_cell(nw[0], nw[1])
            if not np.isfinite(self.length[r, c]):
                return None
        return r, c

    def geodesic(self, xy) -> Optional[float]:
        """Remaining route length in metres (None if the goal is unreachable from xy)."""
        rc = self._cell(xy)
        return None if rc is None else float(self.length[rc])

    def route_cells(self, xy, max_len: Optional[float] = None) -> List[int]:
        """Flat cell indices along the route from xy towards the goal (first = xy's cell)."""
        rc = self._cell(xy)
        if rc is None:
            return []
        W, par = self.W, self._parent_l
        cur = rc[0] * W + rc[1]
        out, acc = [cur], 0.0
        while True:
            p = par[cur]
            if p < 0:
                break
            pr, pc = divmod(p, W)
            cr, cc = divmod(cur, W)
            acc += self.cell * (SQ2 if (pr != cr and pc != cc) else 1.0)
            if max_len is not None and acc > max_len + 1e-9:
                break
            out.append(p)
            cur = p
        return out

    def _los(self, a: int, b: int) -> bool:
        ar, ac = divmod(a, self.W)
        br, bc = divmod(b, self.W)
        n = 2 * max(abs(br - ar), abs(bc - ac)) + 1
        rs = np.rint(np.linspace(ar, br, n)).astype(int)
        cs = np.rint(np.linspace(ac, bc, n)).astype(int)
        return bool(self.free[rs, cs].all())

    def waypoint(self, xy, lookahead: float, los: bool = True) -> Optional[np.ndarray]:
        """Point `lookahead` metres along the route (pulled back to line of sight if `los`)."""
        cells = self.route_cells(xy, max_len=lookahead)
        if not cells:
            return None
        if len(cells) == 1:
            return self._xy(cells[0])
        k = len(cells) - 1
        if los:
            while k > 1 and not self._los(cells[0], cells[k]):
                k -= 1
        return self._xy(cells[k])

    def route_xy(self, xy) -> np.ndarray:
        return np.array([self._xy(c) for c in self.route_cells(xy)])

    def _xy(self, flat: int) -> np.ndarray:
        r, c = divmod(flat, self.W)
        return np.array([(c + 0.5) * self.cell, (r + 0.5) * self.cell], dtype=float)


_FIELDS: Dict[tuple, CostField] = {}
_COSTS: Dict[tuple, np.ndarray] = {}


class Planner:
    """Per-floor-plan planner with a process-wide field cache."""

    def __init__(self, fp: FloorPlan, ratio: float = 0.0) -> None:
        self.fp, self.ratio = fp, float(ratio)
        self._key = (str(getattr(fp, "_path", id(fp))), self.ratio)

    def in_window(self, tod_hour: float) -> bool:
        return bool(self.fp.is_prayer_window(tod_hour)) if self.ratio > 0 else False

    def field(self, goal_xy, tod_hour: float = 12.0) -> CostField:
        win = self.in_window(tod_hour)
        gr, gc = self.fp._world_to_cell(goal_xy[0], goal_xy[1])
        key = self._key + (gr, gc, win)
        f = _FIELDS.get(key)
        if f is None:
            ck = self._key + (win,)
            if ck not in _COSTS:
                _COSTS[ck] = cost_grid(self.fp, self.ratio, win)
            f = _FIELDS[key] = CostField(self.fp, _COSTS[ck], goal_xy)
        return f
