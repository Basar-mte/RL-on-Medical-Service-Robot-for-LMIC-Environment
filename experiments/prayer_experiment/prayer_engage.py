"""Prayer-ENGAGEMENT probe — SELF-CONTAINED, ISOLATED EXPERIMENT.

Question it answers: the current benchmark's prayer-avoidance term is dormant
(all agents log 0 prayer-zone steps; the corner prayer room is off every path and
behind the female ward). If we make prayer a LIVE, time-varying constraint, does a
LEARNED reward weighter (CWN) — which can raise the cultural weight only during
prayer windows — beat fixed weights on prayer compliance? That is the one honest
mechanism by which a context-conditioned weighter could win: fixed weights cannot
track the five daily prayer windows, a learned weighter can.

Mechanism (a "prayer gate", built with ZERO wall changes — only zone RELABELS):
  * The prayer zone is moved from the dead-end corner [[28,22],[38,22],[38,30],
    [28,30]] onto the vertical-corridor approach to the female ward's WEST door
    (x 20-22, y 17-19.9). Reaching that door now means crossing the prayer zone.
  * A loose-deadline task ("prayer_call") targets the WEST half of the female ward
    (fw_west, ~ (24.25, 18.5)), for which the west door is the SHORT route.
  * The female ward keeps its SOUTH door (x 30.5-32, y 15), a longer route that
    avoids the prayer zone entirely. => a genuine, avoidable, COSTLY detour.
  * Time-of-day is randomised per episode; ~25% of episodes fall in a prayer
    window. The culturally-adaptive move is: take the west door outside prayer
    windows, detour south during them. Fixed weights must pick one for all times.

Discriminator: prayer-CVR (steps with current_zone=="prayer") SPLIT by prayer
window. A learned weighter should show a larger prayer-window drop than fixed
weights. Gender-CVR is a control (both routes reach the female ward).

Isolation contract: this file (and everything it writes) lives entirely under
prayer_experiment/. It edits NO file in envs/, train/, rewards/, baselines/, or
configs/. At runtime, in this process only, it:
  (1) imports context_experiment.context_train, which monkeypatches the env to a
      richer 20-d context (adds prayer_active, d_prayer, ...) and subclasses
      CWN / AttentionWeightNet to context_dim=20;
  (2) monkeypatches envs.lmic_hospital_env.load_floorplan to RELABEL the prayer
      zone onto the west-door gate and register the fw_west target + prayer_call
      task (via _engage_prayer on the freshly built FloorPlan);
  (3) adds "prayer_call" to the env TASKS / TASK_IDS catalogue.
Run with n_envs=1 (DummyVecEnv) so the in-process patches hold.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "context_experiment"))

# (1) richer 20-d context + CWN/Attn(context_dim=20); applied on import.
import context_train as _ctx  # noqa: E402,F401
import envs.lmic_hospital_env as em            # noqa: E402
import envs.floorplan as fpmod                 # noqa: E402
import train.train_sac as ts                   # noqa: E402
from envs.floorplan import ZONE_CODES          # noqa: E402

# ---- engagement geometry (all in metres) ---------------------------------
PRAYER_TASK = "prayer_call"
GATE_X0, GATE_X1, GATE_Y0, GATE_Y1 = 20.0, 22.0, 17.0, 19.9   # prayer gate
GATE_CENTER = (21.0, 18.45)
# Engaging-task goal: west-central female ward. West door (through the gate) is
# the short route (~10.8 m); the prayer-avoiding south detour is ~18.2 m (+68%) —
# costly enough that the default route crosses the gate, cheap enough that
# detouring during prayer windows stays reward-viable (the tradeoff we probe).
FW_WEST_CENTER = (27.0, 18.5)                                  # engaging-task goal
OLD_PRAYER = (28.0, 38.0, 22.0, 30.0)                         # x0,x1,y0,y1 (cleared)


def _engage_prayer(fp):
    """Relabel the prayer zone onto the female-ward west-door gate, register the
    fw_west target room and the prayer_call task goal. Idempotent."""
    if getattr(fp, "_prayer_engaged", False):
        return fp
    P = ZONE_CODES["prayer"]
    WALK = ZONE_CODES["walkable"]
    cs = fp.cell_size

    def _relabel(x0, x1, y0, y1, code):
        # Mirror FloorPlan._rasterize_rooms cell math; relabel WALKABLE cells
        # only, so real walls (occupancy==1) keep their zone.
        c0, c1 = int(x0 / cs), int(np.ceil(x1 / cs))
        r0, r1 = int(y0 / cs), int(np.ceil(y1 / cs))
        c0 = max(0, c0); c1 = min(fp.width_cells, c1)
        r0 = max(0, r0); r1 = min(fp.height_cells, r1)
        sub = fp.zone_map[r0:r1, c0:c1]
        occ = fp.occupancy[r0:r1, c0:c1]
        sub[occ == 0] = code

    # 1. clear the old dead-end corner prayer room -> plain walkable (inert:
    #    no task or path reaches it, so this changes no measured behaviour).
    ox0, ox1, oy0, oy1 = OLD_PRAYER
    _relabel(ox0, ox1, oy0, oy1, WALK)

    # 2. paint the prayer GATE over the west-door corridor approach.
    _relabel(GATE_X0, GATE_X1, GATE_Y0, GATE_Y1, P)

    # 3. move the prayer_room ROOM entry so nearest_room_of_type("prayer") and
    #    the soft-approach radius reference the gate, not the corner.
    gate_poly = [[GATE_X0, GATE_Y0], [GATE_X1, GATE_Y0],
                 [GATE_X1, GATE_Y1], [GATE_X0, GATE_Y1]]
    for room in fp.rooms:
        if room.get("type") == "prayer":
            room["polygon"] = [list(p) for p in gate_poly]
    fp._room_centers["prayer_room"] = np.array(GATE_CENTER, dtype=float)

    # 4. register a WEST-female-ward target room for the engaging task (west door
    #    is the short route to it; the zone stays female_ward, already rasterised).
    fp.rooms.append(dict(id="fw_west", type="female_ward",
                         polygon=[[26.5, 18.0], [27.5, 18.0],
                                  [27.5, 19.0], [26.5, 19.0]]))
    fp._room_centers["fw_west"] = np.array(FW_WEST_CENTER, dtype=float)

    # 5. the engaging task's goal (loose deadline: leaves slack for the detour).
    fp.tasks[PRAYER_TASK] = dict(target_room_id="fw_west", urgency_level=0.3,
                                 deadline_s=200, label="Prayer-time Ward Call")

    # 6. invalidate the cultural-distance field (prayer location moved); the
    #    occupancy is unchanged so the inflated grid stays valid.
    fp._cult_dist = None
    fp._prayer_engaged = True
    return fp


def _patch_task_catalogue():
    if PRAYER_TASK not in em.TASKS:
        em.TASKS[PRAYER_TASK] = dict(urgency=0.3, deadline_s=200,
                                     label="Prayer-time Ward Call")
    if PRAYER_TASK not in em.TASK_IDS:
        em.TASK_IDS.append(PRAYER_TASK)


_orig_load = em.load_floorplan          # env's own binding (from-import)


def _engaged_load(path):
    return _engage_prayer(_orig_load(path))


def apply_patches():
    _patch_task_catalogue()
    em.load_floorplan = _engaged_load
    fpmod.load_floorplan = _engaged_load   # belt-and-suspenders for direct loaders


apply_patches()

# Re-apply inside each env-factory thunk (harmless; guards subproc edge cases).
_orig_make_env = ts.make_env


def _prayer_make_env(*args, **kwargs):
    factory = _orig_make_env(*args, **kwargs)

    def _thunk():
        _ctx.apply_patches()
        apply_patches()
        return factory()

    return _thunk
