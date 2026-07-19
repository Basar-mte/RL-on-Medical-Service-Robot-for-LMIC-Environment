"""
Cultural-Adaptive Reward Shaping (CARS) — C2 headline contribution.

Formalises three culturally-grounded spatial constraints as learnable reward terms:

  1. Prayer-space avoidance    – robot must keep distance from prayer room
                                 especially during prayer time windows.
  2. Gender-zone navigation    – robot should minimise time in gender-segregated
                                 wards; when entry is unavoidable, move efficiently.
  3. Attendant-aware proxemics – LMIC family attendants form tight groups;
                                 robot must respect cluster boundaries.

All three terms produce scalar rewards in [-2.0, 0.0] so they compose cleanly
with the navigation and safety rewards.

Reference formulation (playbook §3.4.2):
  R_cultural = α·r_prayer + β·r_gender + γ·r_attendant
  α, β, γ are learned by the CWN (rewards/cwn.py) or fixed during ablations.
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np

from envs.floorplan import FloorPlan, CULTURAL_ZONE_TYPES


# ---------------------------------------------------------------------------
# 1. Prayer-space avoidance
# ---------------------------------------------------------------------------

def prayer_avoidance_reward(
    robot_pos: np.ndarray,
    fp: FloorPlan,
    tod_hour: float,
    *,
    min_safe_dist_m: float = 3.0,
    hard_entry_penalty: float = -2.0,
    soft_approach_scale: float = 0.4,
) -> Tuple[float, bool]:
    """
    Penalise proximity to prayer room, scaled by prayer time intensity.

    Returns
    -------
    (reward, violation_flag)
    """
    # Distance to nearest prayer room center
    prayer_center = fp.nearest_room_of_type(robot_pos[0], robot_pos[1], "prayer")
    if prayer_center is None:
        return 0.0, False

    dist = float(np.linalg.norm(robot_pos - prayer_center))
    intensity = fp.prayer_intensity(tod_hour)  # [0.3, 1.0]

    # Hard violation: robot is inside the prayer zone
    if fp.get_zone(robot_pos[0], robot_pos[1]) == "prayer":
        violation = True
        reward = hard_entry_penalty * intensity
        return float(np.clip(reward, -2.0, 0.0)), violation

    # Soft approach penalty: within safe distance
    if dist < min_safe_dist_m:
        frac = (min_safe_dist_m - dist) / min_safe_dist_m
        reward = -soft_approach_scale * frac * intensity
        return float(np.clip(reward, -1.0, 0.0)), False

    return 0.0, False


# ---------------------------------------------------------------------------
# 2. Gender-zone navigation
# ---------------------------------------------------------------------------

def gender_zone_reward(
    robot_pos: np.ndarray,
    fp: FloorPlan,
    *,
    gender_entry_penalty: float = -1.0,
    transit_penalty_per_step: float = -0.1,
) -> Tuple[float, bool]:
    """
    Penalise robot presence in gender-segregated zones.

    Hard entry penalty once per entry; sustained transit penalty.
    Returns (reward, violation_flag).
    """
    zone = fp.get_zone(robot_pos[0], robot_pos[1])
    if zone in ("male_ward", "female_ward"):
        # Sustained penalty for dwelling
        return transit_penalty_per_step, True
    return 0.0, False


# ---------------------------------------------------------------------------
# 3. Attendant-aware proxemics
# ---------------------------------------------------------------------------

def attendant_proxemics_reward(
    nearest_family_dist: float,
    *,
    cluster_boundary_m: float = 1.2,
    hard_intrusion_m: float = 0.6,
    hard_penalty: float = -1.5,
    soft_scale: float = 0.4,
) -> Tuple[float, bool]:
    """
    Penalise robot intrusion into family-attendant cluster space.

    LMIC context: family members camp beside patients in tight groups.
    The cluster boundary is larger than standard personal space.

    Returns (reward, violation_flag).
    """
    if nearest_family_dist < hard_intrusion_m:
        return hard_penalty, True

    if nearest_family_dist < cluster_boundary_m:
        frac = (cluster_boundary_m - nearest_family_dist) / (
            cluster_boundary_m - hard_intrusion_m
        )
        reward = -soft_scale * frac
        return float(np.clip(reward, -soft_scale, 0.0)), False

    return 0.0, False


# ---------------------------------------------------------------------------
# Composite cultural reward
# ---------------------------------------------------------------------------

def cultural_reward(
    robot_pos: np.ndarray,
    fp: FloorPlan,
    tod_hour: float,
    nearest_family_dist: float,
    weights: np.ndarray = None,
) -> Tuple[float, bool]:
    """
    Composite of all three cultural reward terms.

    Parameters
    ----------
    robot_pos : (2,) array, world coordinates in metres.
    fp : FloorPlan instance.
    tod_hour : float, current simulated hour [0, 24).
    nearest_family_dist : float, distance to nearest family_attendant in metres.
    weights : (3,) array [α, β, γ], optional. Defaults to [1/3, 1/3, 1/3].

    Returns
    -------
    (reward, any_violation)
    """
    if weights is None:
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    r_prayer, v_prayer = prayer_avoidance_reward(robot_pos, fp, tod_hour)
    r_gender, v_gender = gender_zone_reward(robot_pos, fp)
    r_attendant, v_attendant = attendant_proxemics_reward(nearest_family_dist)

    total = (weights[0] * r_prayer
             + weights[1] * r_gender
             + weights[2] * r_attendant)

    any_violation = v_prayer or v_gender or v_attendant
    return float(np.clip(total, -2.0, 0.0)), any_violation


# ---------------------------------------------------------------------------
# Cultural Violation Rate helper (CVR metric — §4.2 in playbook)
# ---------------------------------------------------------------------------

class CulturalViolationTracker:
    """
    Episode-level tracker for Cultural Violation Rate (CVR).

    CVR = (steps in cultural violation) / (total steps)

    Also separately tracks the three sub-violation types.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.prayer_violations = 0
        self.gender_violations = 0
        self.attendant_violations = 0

    def record(
        self,
        robot_pos: np.ndarray,
        fp: FloorPlan,
        tod_hour: float,
        nearest_family_dist: float,
    ) -> None:
        self.steps += 1
        _, vp = prayer_avoidance_reward(robot_pos, fp, tod_hour)
        _, vg = gender_zone_reward(robot_pos, fp)
        _, va = attendant_proxemics_reward(nearest_family_dist)
        if vp:
            self.prayer_violations += 1
        if vg:
            self.gender_violations += 1
        if va:
            self.attendant_violations += 1

    @property
    def cvr(self) -> float:
        if self.steps == 0:
            return 0.0
        total_viol = (self.prayer_violations
                      + self.gender_violations
                      + self.attendant_violations)
        return total_viol / self.steps

    @property
    def prayer_cvr(self) -> float:
        return self.prayer_violations / max(1, self.steps)

    @property
    def gender_cvr(self) -> float:
        return self.gender_violations / max(1, self.steps)

    @property
    def attendant_cvr(self) -> float:
        return self.attendant_violations / max(1, self.steps)

    def summary(self) -> dict:
        return dict(
            cvr=self.cvr,
            prayer_cvr=self.prayer_cvr,
            gender_cvr=self.gender_cvr,
            attendant_cvr=self.attendant_cvr,
            steps=self.steps,
        )
