"""
Floor plan loader for LMIC-HospitalNav-v1.

Loads a parametric hospital floor plan from YAML, rasterizes it to a 2-D
occupancy grid and a zone map, and provides spatial helpers used by the
Gymnasium environment.

Room zone types
---------------
walkable   – generic walkable space
wall       – impassable
corridor   – high-traffic passage (shift-change penalty applies)
prayer     – culturally sensitive (prayer-space penalty)
female_ward – gender-segregated ward
male_ward  – gender-segregated ward
nurse_station, pharmacy, ward, icu, emergency – functional zones
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# Zone type → integer code
ZONE_CODES: Dict[str, int] = {
    "wall":          0,
    "walkable":      1,
    "corridor":      2,
    "nurse_station": 3,
    "pharmacy":      4,
    "ward":          5,
    "icu":           6,
    "emergency":     7,
    "prayer":        8,
    "male_ward":     9,
    "female_ward":   10,
}
CODE_TO_ZONE = {v: k for k, v in ZONE_CODES.items()}

CULTURAL_ZONE_TYPES = {"prayer", "male_ward", "female_ward"}
WALKABLE_ZONE_CODES = set(ZONE_CODES.values()) - {ZONE_CODES["wall"]}


class FloorPlan:
    """
    Rasterized hospital floor plan loaded from a YAML configuration.

    Attributes
    ----------
    occupancy : np.ndarray (H, W) uint8
        0 = walkable, 1 = wall.
    zone_map : np.ndarray (H, W) uint8
        Integer zone code per cell.
    cell_size : float
        Meters per grid cell.
    width_cells, height_cells : int
    rooms : list[dict]
        Raw room definitions from YAML.
    tasks : dict
        Task definitions keyed by task id.
    """

    def __init__(self, yaml_path: str | Path) -> None:
        self._path = Path(yaml_path)
        cfg = yaml.safe_load(self._path.read_text())

        self.width_m: float = cfg["map"]["width_m"]
        self.height_m: float = cfg["map"]["height_m"]
        self.cell_size: float = cfg["map"].get("cell_size_m", 0.1)

        self.width_cells: int = int(round(self.width_m / self.cell_size))
        self.height_cells: int = int(round(self.height_m / self.cell_size))

        self.rooms: List[dict] = cfg.get("rooms", [])
        self.tasks: dict = cfg.get("tasks", {})
        self.prayer_windows: List[dict] = cfg.get("prayer_windows", [])
        self.shift_change_windows: List[dict] = cfg.get("shift_change_windows", [])
        self.robot_spawn_points: List[List[float]] = cfg.get("robot_spawn_points", [[20.0, 13.5]])
        self.pedestrian_spawn_zones: dict = cfg.get("pedestrian_spawn_zones", {})

        # Build grids
        self.occupancy = np.ones((self.height_cells, self.width_cells), dtype=np.uint8)
        self.zone_map = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)

        self._rasterize_rooms()
        self._rasterize_obstacles(cfg.get("static_obstacles", []))
        self._room_centers: Dict[str, np.ndarray] = self._compute_room_centers()

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        col = max(0, min(self.width_cells - 1, col))
        row = max(0, min(self.height_cells - 1, row))
        return row, col

    def _rasterize_rooms(self) -> None:
        for room in self.rooms:
            zone_type = room.get("type", "walkable")
            code = ZONE_CODES.get(zone_type, ZONE_CODES["walkable"])
            poly = np.array(room["polygon"], dtype=float)
            xs, ys = poly[:, 0], poly[:, 1]
            c_min = int(xs.min() / self.cell_size)
            c_max = int(np.ceil(xs.max() / self.cell_size))
            r_min = int(ys.min() / self.cell_size)
            r_max = int(np.ceil(ys.max() / self.cell_size))
            c_min = max(0, c_min); c_max = min(self.width_cells, c_max)
            r_min = max(0, r_min); r_max = min(self.height_cells, r_max)
            self.occupancy[r_min:r_max, c_min:c_max] = 0
            self.zone_map[r_min:r_max, c_min:c_max] = code

    def _rasterize_obstacles(self, obstacles: list) -> None:
        for obs in obstacles:
            poly = np.array(obs["polygon"], dtype=float)
            xs, ys = poly[:, 0], poly[:, 1]
            c_min = max(0, int(xs.min() / self.cell_size))
            c_max = min(self.width_cells, int(np.ceil(xs.max() / self.cell_size)))
            r_min = max(0, int(ys.min() / self.cell_size))
            r_max = min(self.height_cells, int(np.ceil(ys.max() / self.cell_size)))
            self.occupancy[r_min:r_max, c_min:c_max] = 1
            self.zone_map[r_min:r_max, c_min:c_max] = ZONE_CODES["wall"]

    def _compute_room_centers(self) -> Dict[str, np.ndarray]:
        centers = {}
        for room in self.rooms:
            poly = np.array(room["polygon"], dtype=float)
            cx = poly[:, 0].mean()
            cy = poly[:, 1].mean()
            centers[room["id"]] = np.array([cx, cy], dtype=float)
        return centers

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def is_walkable(self, x: float, y: float) -> bool:
        """Return True if world position (x, y) is walkable."""
        if x < 0 or y < 0 or x >= self.width_m or y >= self.height_m:
            return False
        r, c = self._world_to_cell(x, y)
        return bool(self.occupancy[r, c] == 0)

    def get_zone(self, x: float, y: float) -> str:
        """Return zone type string at world position (x, y)."""
        r, c = self._world_to_cell(x, y)
        return CODE_TO_ZONE.get(int(self.zone_map[r, c]), "wall")

    def get_zone_code(self, x: float, y: float) -> int:
        r, c = self._world_to_cell(x, y)
        return int(self.zone_map[r, c])

    def is_cultural_zone(self, x: float, y: float) -> bool:
        return self.get_zone(x, y) in CULTURAL_ZONE_TYPES

    def nearest_walkable(self, x: float, y: float, max_radius: int = 30) -> np.ndarray:
        """Return (x, y) of the nearest walkable cell to the given world pos."""
        if self.is_walkable(x, y):
            return np.array([x, y], dtype=float)
        r0, c0 = self._world_to_cell(x, y)
        for radius in range(1, max_radius):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < self.height_cells and 0 <= c < self.width_cells:
                        if self.occupancy[r, c] == 0:
                            return np.array(
                                [(c + 0.5) * self.cell_size,
                                 (r + 0.5) * self.cell_size], dtype=float)
        return np.array([x, y], dtype=float)

    def room_center(self, room_id: str) -> np.ndarray:
        """Return world-space center of a room."""
        if room_id in self._room_centers:
            return self._room_centers[room_id].copy()
        raise KeyError(f"Room '{room_id}' not found in floor plan.")

    def nearest_room_of_type(self, x: float, y: float, room_type: str) -> Optional[np.ndarray]:
        """Return center of the nearest room of the given type."""
        best_dist = float("inf")
        best_center = None
        for room in self.rooms:
            if room.get("type") == room_type:
                c = self._room_centers[room["id"]]
                dist = np.hypot(c[0] - x, c[1] - y)
                if dist < best_dist:
                    best_dist = dist
                    best_center = c.copy()
        return best_center

    def random_walkable_point(
        self, rng: np.random.Generator,
        zone_type: Optional[str] = None,
        margin: float = 0.5,
    ) -> np.ndarray:
        """Return a random walkable world position, optionally filtered by zone type."""
        code_filter = None if zone_type is None else ZONE_CODES.get(zone_type)
        margin_cells = int(margin / self.cell_size)
        while True:
            r = rng.integers(margin_cells, self.height_cells - margin_cells)
            c = rng.integers(margin_cells, self.width_cells - margin_cells)
            if self.occupancy[r, c] != 0:
                continue
            if code_filter is not None and self.zone_map[r, c] != code_filter:
                continue
            x = (c + 0.5) * self.cell_size
            y = (r + 0.5) * self.cell_size
            return np.array([x, y], dtype=float)

    def random_point_in_spawn_zone(
        self, rng: np.random.Generator, bbox: List[List[float]]
    ) -> np.ndarray:
        """Return a random walkable point within a bounding box [[x0,y0],[x1,y1]]."""
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        for _ in range(200):
            x = rng.uniform(x0, x1)
            y = rng.uniform(y0, y1)
            if self.is_walkable(x, y):
                return np.array([x, y], dtype=float)
        # Fallback: nearest walkable to center
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return self.nearest_walkable(cx, cy)

    # ------------------------------------------------------------------
    # A* path planner
    # ------------------------------------------------------------------

    def astar(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        A* planner on the occupancy grid.

        Parameters
        ----------
        start, goal : array-like (x, y) in world coords.

        Returns
        -------
        List of world-space waypoints, or None if no path found.
        """
        r0, c0 = self._world_to_cell(start[0], start[1])
        rg, cg = self._world_to_cell(goal[0], goal[1])

        # Make sure endpoints are walkable
        if self.occupancy[r0, c0]:
            nw = self.nearest_walkable(start[0], start[1])
            r0, c0 = self._world_to_cell(nw[0], nw[1])
        if self.occupancy[rg, cg]:
            nw = self.nearest_walkable(goal[0], goal[1])
            rg, cg = self._world_to_cell(nw[0], nw[1])

        def h(r, c):
            return abs(r - rg) + abs(c - cg)

        open_heap: list = []
        heapq.heappush(open_heap, (h(r0, c0), 0, r0, c0, None))
        came_from: dict = {}
        g_score: dict = {(r0, c0): 0}
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_heap:
            _, g, r, c, parent = heapq.heappop(open_heap)
            if (r, c) in came_from:
                continue
            came_from[(r, c)] = parent
            if r == rg and c == cg:
                # Reconstruct
                path = []
                node = (r, c)
                while node is not None:
                    nr, nc = node
                    path.append(np.array(
                        [(nc + 0.5) * self.cell_size,
                         (nr + 0.5) * self.cell_size], dtype=float))
                    node = came_from[node]
                path.reverse()
                return path
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.height_cells and 0 <= nc < self.width_cells):
                    continue
                if self.occupancy[nr, nc]:
                    continue
                step = 1.414 if (dr != 0 and dc != 0) else 1.0
                ng = g + step
                if ng < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = ng
                    f = ng + h(nr, nc)
                    heapq.heappush(open_heap, (f, ng, nr, nc, (r, c)))
        return None

    def path_length(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Return A* path length in metres, or Euclidean if no path."""
        path = self.astar(start, goal)
        if path is None or len(path) < 2:
            return float(np.linalg.norm(goal - start))
        return sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))

    def validate(self) -> bool:
        """Check all room centers are reachable from the nurse station."""
        try:
            origin = self.room_center("nurse_station")
        except KeyError:
            origin = np.array([20.0, 13.5])
        for room in self.rooms:
            center = self._room_centers[room["id"]]
            path = self.astar(origin, center)
            if path is None:
                print(f"[FloorPlan] WARNING: room '{room['id']}' unreachable from nurse station.")
        return True

    # ------------------------------------------------------------------
    # Time-of-day helpers
    # ------------------------------------------------------------------

    def is_prayer_window(self, hour: float) -> bool:
        for pw in self.prayer_windows:
            if pw["start_hour"] <= hour < pw["end_hour"]:
                return True
        return False

    def prayer_intensity(self, hour: float) -> float:
        """Return scalar in [0.3, 1.0] reflecting prayer-time intensity."""
        for pw in self.prayer_windows:
            if pw["start_hour"] <= hour < pw["end_hour"]:
                return 1.0
        return 0.3

    def is_shift_change(self, hour: float) -> bool:
        for sw in self.shift_change_windows:
            if sw["start_hour"] <= hour < sw["end_hour"]:
                return True
        return False

    def zone_onehot(self, x: float, y: float) -> np.ndarray:
        """
        Return 8-dim one-hot zone encoding used in the context vector.
        Zones: [walkable, corridor, nurse_station, pharmacy, ward/icu/emergency,
                prayer, male_ward, female_ward]
        """
        zone = self.get_zone(x, y)
        vec = np.zeros(8, dtype=np.float32)
        mapping = {
            "walkable": 0, "corridor": 1, "nurse_station": 2,
            "pharmacy": 3, "ward": 4, "icu": 4, "emergency": 4,
            "prayer": 5, "male_ward": 6, "female_ward": 7,
        }
        idx = mapping.get(zone, 0)
        vec[idx] = 1.0
        return vec


# Convenience function --------------------------------------------------

def load_floorplan(path: str | Path) -> FloorPlan:
    """Load and return a FloorPlan from the given YAML path."""
    fp = FloorPlan(path)
    fp.validate()
    return fp
