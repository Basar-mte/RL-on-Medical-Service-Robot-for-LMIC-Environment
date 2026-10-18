"""
Multi-class pedestrian crowd simulation using the Headed Social Force Model (HSFM).

References
----------
Helbing, D. & Molnár, P. (1995). Social force model for pedestrian dynamics.
    Physical Review E, 51(5), 4282-4286. doi:10.1103/PhysRevE.51.4282
Farina, F., Fontanelli, D., Garulli, A., Giannitrapani, A., & Prattichizzo, D. (2017).
    Walking Ahead: The Headed Social Force Model. PLOS ONE, 12(1), e0169734.
    doi:10.1371/journal.pone.0169734

Pedestrian classes (LMIC-specific)
------------------------------------
medical_staff   : 1.2–1.4 m/s, purposeful, small personal space
patient         : 0.5–0.8 m/s, slow + random direction noise
family_attendant: 0.4–0.7 m/s, moves in groups of 2–4 (LMIC-specific),
                  loiters near bed areas and corridors
wheelchair      : 0.6–0.9 m/s, wider footprint, slower turns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# -------------------------------------------------------------------
# Class parameter table
# -------------------------------------------------------------------
# All distances in metres, speeds in m/s, mass in kg, tau in s.
PCLASS_PARAMS: Dict[str, dict] = {
    "medical_staff": dict(
        desired_speed=(1.2, 1.4),
        personal_space=0.4,
        mass=70.0,
        tau=0.5,           # relaxation time
        radius=0.25,       # physical body radius (m)
        direction_noise=0.05,
        group_size=(1, 1),
    ),
    "patient": dict(
        desired_speed=(0.5, 0.8),
        personal_space=0.6,
        mass=65.0,
        tau=0.7,
        radius=0.25,
        direction_noise=0.25,  # frequent direction changes
        group_size=(1, 1),
    ),
    "family_attendant": dict(
        desired_speed=(0.4, 0.7),
        personal_space=0.5,
        mass=68.0,
        tau=0.6,
        radius=0.25,
        direction_noise=0.1,
        group_size=(2, 4),     # LMIC-specific: loiters in groups
        cohesion_force=3.0,    # N
        cohesion_radius=1.8,   # m
    ),
    "wheelchair": dict(
        desired_speed=(0.6, 0.9),
        personal_space=0.7,
        mass=120.0,
        tau=0.8,
        radius=0.45,           # wider footprint
        direction_noise=0.05,
        group_size=(1, 1),
    ),
}

# HSFM force parameters (Helbing & Molnar 1995 + Farina 2017)
# Interaction-force constants. The historical values (A_REPULSE=2.0 N,
# B_REPULSE=0.08 m on CENTER distance) were physically inert against the
# goal-driving force, whose gain is mass/tau ~ 140 N per m/s of velocity
# error for a 70 kg pedestrian: at any separation the repulsion changed a
# pedestrian's velocity by <0.01 m/s per second, so pedestrians walked
# straight through each other and through the robot. Verified empirically:
# a robot standing perfectly still was "collided into" in 60% of episodes.
# The constants below use the standard surface-distance form
# A*exp(-(d - r_i - r_j)/B) + K_BODY*overlap (Helbing-style, scaled to this
# model's masses) so repulsion is commensurate with goal attraction.
A_REPULSE = 60.0   # pedestrian-pedestrian repulsion amplitude (N)
B_REPULSE = 0.25   # decay length (m), measured from surface separation
K_BODY    = 600.0  # contact spring (N per metre of body overlap)
ROBOT_BODY_RADIUS = 0.25  # matches envs.lmic_hospital_env.ROBOT_RADIUS
# Robot term: stronger and longer-ranged than ped-ped — any contact with the
# robot terminates the episode, so pedestrians must start yielding while
# still ~1m away (humans also give mobile robots a wider berth than they
# give each other; cf. HRI proxemics).
A_ROBOT = 200.0    # robot repulsion amplitude (N)
B_ROBOT = 0.5      # decay length (m), measured from surface separation
A_WALL    = 10.0   # wall repulsion amplitude (N)
B_WALL    = 0.1    # wall decay length (m)
GOAL_FORCE_SCALE = 1.5
MAX_SPEED_FACTOR  = 1.5  # max speed = desired * this


@dataclass
class Pedestrian:
    pos: np.ndarray        # (2,) world coords (x, y)
    vel: np.ndarray        # (2,) velocity
    heading: float         # radians
    goal: np.ndarray       # (2,) current waypoint
    radius: float
    pclass: str
    desired_speed: float
    personal_space: float
    mass: float
    tau: float
    direction_noise: float
    group_id: int = -1     # -1 = not in group
    active: bool = True

    def to_state(self) -> np.ndarray:
        """Return (x, y, vx, vy, cos_h, sin_h, radius, class_onehot×4) — 12 dims."""
        classes = ["medical_staff", "patient", "family_attendant", "wheelchair"]
        oh = np.zeros(4, dtype=np.float32)
        if self.pclass in classes:
            oh[classes.index(self.pclass)] = 1.0
        return np.array(
            [self.pos[0], self.pos[1], self.vel[0], self.vel[1],
             np.cos(self.heading), np.sin(self.heading), self.radius,
             *oh], dtype=np.float32)


class PedestrianCrowd:
    """
    Vectorised crowd simulation using HSFM.

    Parameters
    ----------
    floorplan : FloorPlan
        Used for wall repulsion sampling.
    max_pedestrians : int
        Hard cap on simultaneous agents.
    rng : np.random.Generator, optional
        For reproducibility.
    """

    def __init__(self, floorplan, max_pedestrians: int = 40,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.fp = floorplan
        self.max_n = max_pedestrians
        self.rng = rng or np.random.default_rng()
        self.agents: List[Pedestrian] = []
        self._group_counter = 0
        self._wall_field: Optional[np.ndarray] = None  # built lazily per fp

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def reset(self, counts: Dict[str, int], rng: Optional[np.random.Generator] = None,
              avoid_pos: Optional[np.ndarray] = None, avoid_radius: float = 1.5) -> None:
        """
        Re-spawn all pedestrians.

        Parameters
        ----------
        counts : dict  {pclass: n_agents}
        rng    : optional new RNG seed
        avoid_pos : optional (2,) position (the robot spawn) no pedestrian may
            spawn within `avoid_radius` of — prevents instant, policy-
            independent spawn collisions (changelog known gap).
        """
        if rng is not None:
            self.rng = rng
        self._avoid_pos = None if avoid_pos is None else np.asarray(avoid_pos, dtype=float)
        self._avoid_radius = float(avoid_radius)
        self.agents = []
        self._group_counter = 0
        total = sum(counts.values())
        if total > self.max_n:
            # Scale down proportionally
            scale = self.max_n / total
            counts = {k: max(1, int(v * scale)) for k, v in counts.items()}

        for pclass, n in counts.items():
            params = PCLASS_PARAMS[pclass]
            gs_min, gs_max = params["group_size"]
            spawned = 0
            # Bounded attempts: _spawn_one returning None (spawn zones
            # saturated by the 0.8 m inter-agent spacing and the robot
            # avoid-radius) previously left `spawned` unchanged and this
            # loop spun forever — a hard hang inside env.reset().
            attempts = 0
            while spawned < n and attempts < 50 * n + 200:
                attempts += 1
                # Group size
                gs = self.rng.integers(gs_min, gs_max + 1) if gs_max > 1 else 1
                gs = min(gs, n - spawned)
                gid = self._group_counter if gs > 1 else -1
                if gs > 1:
                    self._group_counter += 1
                for _ in range(gs):
                    agent = self._spawn_one(pclass, gid)
                    if agent is not None:
                        self.agents.append(agent)
                        spawned += 1
            if spawned < n:
                print(f"[PedestrianCrowd] WARNING: spawned {spawned}/{n} "
                      f"'{pclass}' agents (spawn space saturated)")

    def _spawn_one(self, pclass: str, group_id: int) -> Optional[Pedestrian]:
        params = PCLASS_PARAMS[pclass]
        spawn_zones = self.fp.pedestrian_spawn_zones.get(pclass, [])

        for _ in range(100):
            if spawn_zones:
                bbox = spawn_zones[self.rng.integers(len(spawn_zones))]
                pos = self.fp.random_point_in_spawn_zone(self.rng, bbox)
            else:
                pos = self.fp.random_walkable_point(self.rng)

            # Avoid spawning too close to other agents
            too_close = any(
                np.linalg.norm(a.pos - pos) < 0.8
                for a in self.agents
            )
            if too_close:
                continue

            # Avoid the robot's spawn area (see reset(avoid_pos=...))
            if (getattr(self, "_avoid_pos", None) is not None
                    and np.linalg.norm(pos - self._avoid_pos) < self._avoid_radius):
                continue

            # Pick a random goal from a different room/area
            goal = self._pick_goal(pos, pclass)
            speed = self.rng.uniform(*params["desired_speed"])
            heading = self.rng.uniform(0, 2 * np.pi)

            return Pedestrian(
                pos=pos.copy(),
                vel=np.array([np.cos(heading) * 0.1, np.sin(heading) * 0.1]),
                heading=heading,
                goal=goal,
                radius=params["radius"],
                pclass=pclass,
                desired_speed=speed,
                personal_space=params["personal_space"],
                mass=params["mass"],
                tau=params["tau"],
                direction_noise=params["direction_noise"],
                group_id=group_id,
            )
        return None

    def _pick_goal(self, pos: np.ndarray, pclass: str) -> np.ndarray:
        """Pick a goal position appropriate for the pedestrian class."""
        for _ in range(50):
            goal = self.fp.random_walkable_point(self.rng)
            if np.linalg.norm(goal - pos) > 3.0:
                return goal
        return self.fp.random_walkable_point(self.rng)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, dt: float = 0.1, robot_pos: Optional[np.ndarray] = None) -> None:
        """Advance all pedestrians by dt seconds using semi-implicit Euler."""
        n = len(self.agents)
        if n == 0:
            return

        # Stack positions and velocities for vectorised ops
        pos = np.stack([a.pos for a in self.agents])         # (n, 2)
        vel = np.stack([a.vel for a in self.agents])         # (n, 2)
        goals = np.stack([a.goal for a in self.agents])      # (n, 2)
        desired = np.array([a.desired_speed for a in self.agents])  # (n,)
        masses = np.array([a.mass for a in self.agents])     # (n,)
        taus = np.array([a.tau for a in self.agents])        # (n,)
        radii = np.array([a.radius for a in self.agents])    # (n,)
        group_ids = np.array([a.group_id for a in self.agents])  # (n,)

        # --- Goal attraction (HSFM heading drive) ---
        goal_dir = goals - pos  # (n, 2)
        goal_dist = np.linalg.norm(goal_dir, axis=1, keepdims=True) + 1e-8
        goal_dir_norm = goal_dir / goal_dist
        desired_vel = goal_dir_norm * desired[:, None]
        goal_force = masses[:, None] / taus[:, None] * (desired_vel - vel)
        goal_force *= GOAL_FORCE_SCALE

        # --- Pedestrian-pedestrian repulsion (anisotropic, heading-aware) ---
        diff = pos[:, None, :] - pos[None, :, :]   # (n, n, 2)
        dist_mat = np.linalg.norm(diff, axis=2)    # (n, n)
        np.fill_diagonal(dist_mat, 1e8)
        ped_force = np.zeros_like(pos)
        for i, a in enumerate(self.agents):
            dvec = pos[i] - pos                    # (n, 2) from others to i
            d = dist_mat[:, i]                     # (n,)
            # Repulsion from close pedestrians
            mask = (d < 4.0)
            mask[i] = False
            if mask.any():
                directions = dvec[mask] / (d[mask, None] + 1e-8)
                overlap = radii[i] + radii[mask] - d[mask]  # >0 = touching
                # Farina 2017: anisotropic factor (front-weighted repulsion).
                # `directions` points FROM the other pedestrian TO agent i,
                # so an other IN FRONT of i has directions . goal_dir < 0 —
                # hence the minus sign (0.5 + 0.5*cos_phi weighted people
                # BEHIND more strongly, the exact inverse of the model).
                cos_phi = np.einsum('ij,j->i', directions, goal_dir_norm[i])
                lambda_factor = 0.5 - 0.5 * cos_phi  # [0,1], 1 = dead ahead
                # Surface-distance exponential + contact spring (see constants
                # block above for why the old center-distance form was inert).
                strength = (A_REPULSE * np.exp(np.minimum(overlap, 0.0) / B_REPULSE)
                            + K_BODY * np.maximum(overlap, 0))
                ped_force[i] += ((strength * lambda_factor)[:, None] * directions).sum(axis=0)

        # --- Robot repulsion (pedestrians see and avoid the robot) ---
        # robot_pos was previously accepted but never used: pedestrians were
        # blind to the robot and walked straight through its footprint, and
        # each such contact terminated the episode as a "collision" that no
        # policy could avoid (measured consequence: trained agents scored the
        # same success rate as a random policy). Same functional form as the
        # pedestrian-pedestrian term above, with the robot as one more body.
        if robot_pos is not None:
            dvec_r = pos - np.asarray(robot_pos, dtype=float)[None, :]
            d_r = np.linalg.norm(dvec_r, axis=1)
            mask_r = d_r < 4.0
            if mask_r.any():
                dirs_r = dvec_r[mask_r] / (d_r[mask_r, None] + 1e-8)
                surf_r = d_r[mask_r] - (radii[mask_r] + ROBOT_BODY_RADIUS)
                strength_r = A_ROBOT * np.exp(-np.maximum(surf_r, 0.0) / B_ROBOT) \
                             + K_BODY * np.maximum(-surf_r, 0)
                ped_force[mask_r] += strength_r[:, None] * dirs_r

        # --- Wall repulsion (precomputed per-cell field) ---
        # Replaces the per-agent Python loop of _wall_force (45 agents x 8
        # probes x is_walkable per step, ~41% of crowd.step wall time) with
        # one vectorized lookup into a field built once per floor plan. The
        # field replicates _wall_force at cell-centre resolution (0.1 m).
        if self._wall_field is None:
            self._build_wall_force_field()
        cell = self.fp.cell_size
        rows_p = np.clip((pos[:, 1] / cell).astype(int), 0, self.fp.height_cells - 1)
        cols_p = np.clip((pos[:, 0] / cell).astype(int), 0, self.fp.width_cells - 1)
        wall_force = self._wall_field[rows_p, cols_p].astype(float)

        # --- Group cohesion force (family_attendant only) ---
        cohesion_force = np.zeros_like(pos)
        for gid in np.unique(group_ids[group_ids >= 0]):
            members = np.where(group_ids == gid)[0]
            if len(members) < 2:
                continue
            center = pos[members].mean(axis=0)
            params = PCLASS_PARAMS["family_attendant"]
            c_strength = params["cohesion_force"]
            c_radius = params["cohesion_radius"]
            for i in members:
                diff_c = center - pos[i]
                dist_c = np.linalg.norm(diff_c)
                if dist_c > c_radius:
                    cohesion_force[i] += c_strength * diff_c / (dist_c + 1e-8)

        # --- Total force and integrate ---
        total_force = goal_force + ped_force + wall_force + cohesion_force

        # Add directional noise
        for i, a in enumerate(self.agents):
            noise_angle = self.rng.normal(0, a.direction_noise)
            rot = np.array([[np.cos(noise_angle), -np.sin(noise_angle)],
                            [np.sin(noise_angle),  np.cos(noise_angle)]])
            total_force[i] = rot @ total_force[i]

        # Semi-implicit Euler
        acc = total_force / masses[:, None]
        new_vel = vel + acc * dt
        # Clip to max speed
        speeds = np.linalg.norm(new_vel, axis=1, keepdims=True)
        max_speed = desired * MAX_SPEED_FACTOR
        new_vel = np.where(
            speeds > max_speed[:, None],
            new_vel / (speeds + 1e-8) * max_speed[:, None],
            new_vel,
        )
        new_pos = pos + new_vel * dt

        # Clamp to walkable space
        for i in range(n):
            if not self.fp.is_walkable(new_pos[i, 0], new_pos[i, 1]):
                new_vel[i] = -vel[i] * 0.5  # bounce back
                new_pos[i] = pos[i]

        # Clamp out of the robot's footprint (position-level guarantee, like
        # the wall clamp above): even in a dense jam where body-contact
        # springs push pedestrians around, a pedestrian can never step into
        # contact range of the robot. The episode-terminating collision check
        # therefore only ever fires when the ROBOT moves into a pedestrian —
        # i.e. collisions are attributable to the policy, not to the crowd.
        if robot_pos is not None:
            rp = np.asarray(robot_pos, dtype=float)
            for i in range(n):
                clearance = ROBOT_BODY_RADIUS + radii[i] + 0.02
                if np.linalg.norm(new_pos[i] - rp) < clearance:
                    if np.linalg.norm(pos[i] - rp) >= clearance:
                        new_pos[i] = pos[i]      # stay put rather than enter
                        new_vel[i] = vel[i] * 0.0
                    else:
                        # Already inside (e.g. robot drove in): push straight
                        # out to the clearance ring so the overlap resolves.
                        away = pos[i] - rp
                        nrm = np.linalg.norm(away) + 1e-8
                        new_pos[i] = rp + away / nrm * clearance
                        new_vel[i] = vel[i] * 0.0

        # Update agents
        for i, a in enumerate(self.agents):
            a.pos = new_pos[i]
            a.vel = new_vel[i]
            spd = np.linalg.norm(new_vel[i])
            if spd > 1e-3:
                a.heading = np.arctan2(new_vel[i, 1], new_vel[i, 0])
            # Check goal reached → pick new goal
            if np.linalg.norm(a.pos - a.goal) < 1.0:
                a.goal = self._pick_goal(a.pos, a.pclass)
            # Stall recovery: _pick_goal has no reachability check, so an
            # agent whose goal lies behind a wall pushes into it forever and
            # the crowd gradually freezes solid. 10 s without meaningful
            # motion → resample the goal.
            if spd < 0.05:
                a._stall_steps = getattr(a, "_stall_steps", 0) + 1
                if a._stall_steps > 100:
                    a.goal = self._pick_goal(a.pos, a.pclass)
                    a._stall_steps = 0
            else:
                a._stall_steps = 0

    def _build_wall_force_field(self) -> None:
        """
        Precompute the per-cell wall-repulsion vector: for every grid cell,
        the sum over its 8 neighbours of A_WALL * exp(-dist/B_WALL) pointing
        away from each wall neighbour — exactly what _wall_force computes
        per agent per step, evaluated once at cell-centre resolution.
        """
        occ = self.fp.occupancy.astype(bool)
        H, W = occ.shape
        cell = self.fp.cell_size
        field = np.zeros((H, W, 2), dtype=np.float32)
        for dxs in (-1, 0, 1):
            for dys in (-1, 0, 1):
                if dxs == 0 and dys == 0:
                    continue
                hyp = float(np.hypot(dxs, dys))
                dist = hyp * cell
                strength = A_WALL * float(np.exp(-dist / B_WALL))
                dir_x, dir_y = -dxs / hyp, -dys / hyp
                # shifted[r, c] = occ[r + dys, c + dxs] (True = wall neighbour)
                shifted = np.zeros_like(occ)
                shifted[max(0, -dys):H - max(0, dys), max(0, -dxs):W - max(0, dxs)] = \
                    occ[max(0, dys):H - max(0, -dys), max(0, dxs):W - max(0, -dxs)]
                field[shifted, 0] += strength * dir_x
                field[shifted, 1] += strength * dir_y
        self._wall_field = field

    def _wall_force(self, pos: np.ndarray, radius: float) -> np.ndarray:
        """Approximate wall repulsion by probing cells around the agent."""
        force = np.zeros(2, dtype=float)
        cell = self.fp.cell_size
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = pos[0] + dx * cell
                ny = pos[1] + dy * cell
                if not self.fp.is_walkable(nx, ny):
                    dist = np.sqrt(dx**2 + dy**2) * cell
                    if dist < (radius + 0.3):
                        direction = np.array([-dx, -dy], dtype=float)
                        norm = np.linalg.norm(direction) + 1e-8
                        strength = A_WALL * np.exp(-dist / B_WALL)
                        force += strength * direction / norm
        return force

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_states(self) -> np.ndarray:
        """Return (n, 12) state array for all active agents."""
        if not self.agents:
            return np.zeros((0, 12), dtype=np.float32)
        return np.stack([a.to_state() for a in self.agents])

    def get_local_neighbors(
        self, robot_xy: np.ndarray, k: int = 5, radius: float = 5.0
    ) -> np.ndarray:
        """
        Return relative (dx, dy, vx, vy) for the k nearest pedestrians
        within `radius` metres.  Shape: (k, 4).  Pads zeros if fewer than k.
        """
        result = np.zeros((k, 4), dtype=np.float32)
        if not self.agents:
            return result
        dists = [np.linalg.norm(a.pos - robot_xy) for a in self.agents]
        close = [(d, i) for i, d in enumerate(dists) if d <= radius]
        close.sort()
        for slot, (d, idx) in enumerate(close[:k]):
            a = self.agents[idx]
            result[slot, 0] = a.pos[0] - robot_xy[0]
            result[slot, 1] = a.pos[1] - robot_xy[1]
            result[slot, 2] = a.vel[0]
            result[slot, 3] = a.vel[1]
        return result

    def crowd_density(self, robot_xy: np.ndarray, radius: float = 5.0) -> float:
        """Fraction of pedestrians within `radius` m, normalised to [0, 1]."""
        if not self.agents:
            return 0.0
        n_close = sum(1 for a in self.agents if np.linalg.norm(a.pos - robot_xy) <= radius)
        max_density = min(self.max_n, 40)
        return min(1.0, n_close / max(1, max_density * 0.25))

    def nearest_family_distance(self, robot_xy: np.ndarray) -> float:
        """Distance to nearest family_attendant, or 10.0 if none present."""
        dists = [
            np.linalg.norm(a.pos - robot_xy)
            for a in self.agents if a.pclass == "family_attendant"
        ]
        return min(dists) if dists else 10.0

    @property
    def n_agents(self) -> int:
        return len(self.agents)
