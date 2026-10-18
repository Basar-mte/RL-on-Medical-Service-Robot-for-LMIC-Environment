"""
CARS reward components (C2 headline contribution).

Each function returns a scalar float in a bounded range.
All components are composited in CARS:
  R = w_nav * r_nav + w_safety * r_safety + w_social * r_cult + w_task * r_task

Bounds (approximate):
  r_nav     [-1.0,  1.0]   dense shaping via potential-based guidance
  r_safety  [-2.0,  0.0]   penalty only
  r_task    [-1.0,  2.0]   time bonus + completion bonus
"""

from __future__ import annotations
import math
import numpy as np

# Multiplier on per-step goal progress (see navigation_reward docstring).
# Shared by every reward implementation (CARS here, He2025RewardFn,
# VanillaSACReward) so agents stay comparable.
PROGRESS_SCALE = 10.0


# ---------------------------------------------------------------------------
# Navigation reward  (progress shaping)
# ---------------------------------------------------------------------------

def navigation_reward(
    prev_dist: float,
    curr_dist: float,
    task_complete: bool,
    action: np.ndarray,
    *,
    gamma: float = 0.99,
    goal_bonus: float = 5.0,
    timeout_penalty: float = -0.05,
    backward_penalty: float = -0.02,
) -> float:
    """
    Progress-based navigation shaping Φ(s) = –d(s, goal).

    r_nav = Φ(s') – Φ(s) + goal_bonus·[task_complete]

    Deliberately NOT the discounted form γ·Φ(s')–Φ(s): with Φ = –d and
    γ = 0.99 that form leaks a per-step bonus of (1–γ)·d ≈ +0.01·d for
    merely EXISTING far from the goal — at d = 15 m that is +0.15/step,
    which over a long survived episode dwarfs the one-off goal bonus.
    Measured consequence before this fix: SAC converged to "survive and
    loiter far from the goal" (episode length 700+, return ~16, success
    rate exactly 0 at 80k steps). The undiscounted difference telescopes
    to d_start – d_end over the episode, so wandering is reward-neutral
    and only genuine approach pays. (gamma is kept in the signature for
    API compatibility but intentionally unused.)

    PROGRESS_SCALE: raw per-step progress is at most v_max·Δt = 0.08 m —
    an order of magnitude below the ambient safety/cultural proximity
    penalties accrued while crossing an LMIC-density ward (measured: with
    unscaled progress the return of a completed 20 m task was NET NEGATIVE
    versus idling in an empty corner, and success rate stayed exactly 0).
    Scaling progress so a full-speed approach step ≈ +0.8 makes purposeful
    navigation dominate ambient penalties while zone penalties (−2/step
    inside an active prayer window) remain decisively binding.
    """
    phi_prev = -prev_dist
    phi_curr = -curr_dist
    r = PROGRESS_SCALE * (phi_curr - phi_prev)

    if task_complete:
        r += goal_bonus

    v = float(action[0])
    # Idle penalty on |v|: reverse is now a legal action (the physical
    # chassis can back up), so "idle" means near-zero speed in either
    # direction — a bare v < 0.01 would penalize every reverse maneuver as
    # idling on top of the explicit backward penalty below.
    if abs(v) < 0.01:
        r += timeout_penalty

    # Small backward penalty (reverse is allowed but shouldn't be preferred)
    if v < 0:
        r += backward_penalty

    return float(np.clip(r, -2.0, 6.0))


# ---------------------------------------------------------------------------
# Safety reward
# ---------------------------------------------------------------------------

def safety_reward(
    collision: bool,
    min_ped_dist: float,
    *,
    collision_penalty: float = -5.0,
    personal_space_m: float = 0.8,
    comfort_zone_m: float = 1.5,
    close_penalty_scale: float = 0.5,
) -> float:
    """
    Penalise robot collisions and proxemic space violations.

    Hard collision: -5.0
    Inside personal space (< 0.8m): linear penalty capped at -1.0
    Inside comfort zone (0.8–1.5m): soft linear penalty
    """
    if collision:
        return collision_penalty

    if min_ped_dist < personal_space_m:
        frac = (personal_space_m - min_ped_dist) / personal_space_m
        return float(np.clip(-close_penalty_scale * frac, -1.0, 0.0))

    if min_ped_dist < comfort_zone_m:
        frac = (comfort_zone_m - min_ped_dist) / (comfort_zone_m - personal_space_m)
        return float(-0.05 * frac)

    return 0.0


# ---------------------------------------------------------------------------
# Task reward
# ---------------------------------------------------------------------------

def task_reward(
    task_complete: bool,
    urgency: float,
    elapsed_s: float,
    deadline_s: float,
    *,
    base_bonus: float = 2.0,
    time_penalty_scale: float = 0.5,
) -> float:
    """
    Reward task completion scaled by urgency; penalise time over deadline.

      r_task = urgency * base_bonus            [if completed on time]
             = urgency * base_bonus * 0.5      [if completed late]
             = -urgency * time_penalty_scale * (elapsed/deadline - 1)  [ongoing late]
    """
    if task_complete:
        if elapsed_s <= deadline_s:
            return float(urgency * base_bonus)
        else:
            return float(urgency * base_bonus * 0.5)

    if elapsed_s > deadline_s:
        overshoot = (elapsed_s / deadline_s) - 1.0
        return float(np.clip(-urgency * time_penalty_scale * overshoot, -1.5, 0.0))

    return 0.0


# ---------------------------------------------------------------------------
# Progress reward (auxiliary, used in ablations)
# ---------------------------------------------------------------------------

def progress_reward(
    prev_dist: float,
    curr_dist: float,
    scale: float = 0.3,
) -> float:
    """Simple signed progress toward goal."""
    return float(np.clip((prev_dist - curr_dist) * scale, -0.5, 0.5))


# ---------------------------------------------------------------------------
# Smoothness reward (penalise high angular velocity jerk)
# ---------------------------------------------------------------------------

def smoothness_reward(
    action: np.ndarray,
    prev_action: np.ndarray,
    *,
    angular_jerk_scale: float = 0.02,
) -> float:
    """Penalise large changes in angular velocity."""
    delta_omega = float(abs(action[1] - prev_action[1]))
    return float(-angular_jerk_scale * delta_omega)
