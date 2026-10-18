"""Validate the prayer-engagement patch BEFORE spending any GPU time.

Checks, on the engaged floor plan (built through the monkeypatched loader):
  1. gate painted   : get_zone(gate center) == "prayer"
  2. corner cleared : get_zone(old corner) != "prayer"
  3. prayer room ref : nearest_room_of_type("prayer") == gate center
  4. reach via WEST  : shortest path spawn -> fw_west crosses the prayer gate
  5. reach via SOUTH : with the gate blocked, a path STILL exists (detour works)
  6. detour cost     : south-route length vs west-route length (the tradeoff)
  7. no regression   : every original task goal still reachable from the spawn
Nothing is trained; this only reads the patched geometry.

Run:  .venv\\Scripts\\python.exe prayer_experiment\\validate_prayer.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "prayer_experiment"))

import prayer_engage as pe   # noqa: E402  (applies all patches on import)
import envs.lmic_hospital_env as em  # noqa: E402

CONFIG = str(ROOT / "configs" / "dmch_hospital.yaml")
SPAWN = (20.0, 13.5)


def _crosses_gate(path):
    if path is None:
        return None
    n = 0
    for (x, y) in path:
        if pe.GATE_X0 <= x <= pe.GATE_X1 and pe.GATE_Y0 <= y <= pe.GATE_Y1:
            n += 1
    return n


def main():
    fp = em.load_floorplan(CONFIG)   # engaged loader
    ok = True

    # 1-3: zone bookkeeping
    z_gate = fp.get_zone(*pe.GATE_CENTER)
    z_corner = fp.get_zone(33.0, 26.0)
    pc = fp.nearest_room_of_type(*SPAWN, "prayer")
    print(f"[1] gate zone at {pe.GATE_CENTER}  = {z_gate!r}   "
          f"({'OK' if z_gate == 'prayer' else 'FAIL'})")
    print(f"[2] old corner zone at (33,26)     = {z_corner!r}   "
          f"({'OK' if z_corner != 'prayer' else 'FAIL'})")
    print(f"[3] nearest prayer room center     = {None if pc is None else tuple(np.round(pc,2))}   "
          f"({'OK' if pc is not None and abs(pc[0]-pe.GATE_CENTER[0])<1 and abs(pc[1]-pe.GATE_CENTER[1])<1 else 'FAIL'})")
    ok &= (z_gate == "prayer") and (z_corner != "prayer") and (pc is not None)

    # 4: WEST route (unconstrained shortest path) — should cross the gate
    goal = fp.nearest_walkable(*pe.FW_WEST_CENTER, grid=fp.occupancy_inflated)
    west = fp.astar(np.array(SPAWN), goal)
    west_len = fp.path_length(np.array(SPAWN), goal)
    west_cross = _crosses_gate(west)
    print(f"[4] WEST route spawn->fw_west      : reachable={west is not None}  "
          f"len={west_len:.1f}m  gate-cells-on-path={west_cross}   "
          f"({'OK' if west and west_cross else 'FAIL'})")
    ok &= bool(west) and bool(west_cross)

    # 5-6: SOUTH detour — block the gate, confirm a path still exists, measure cost
    occ_saved = fp.occupancy.copy()
    inf_saved = fp._occ_inflated
    c0, c1 = int(pe.GATE_X0/fp.cell_size), int(np.ceil(pe.GATE_X1/fp.cell_size))
    r0, r1 = int(pe.GATE_Y0/fp.cell_size), int(np.ceil(pe.GATE_Y1/fp.cell_size))
    fp.occupancy[r0:r1, c0:c1] = 1        # temporarily wall off the gate
    fp._occ_inflated = None               # force inflated-grid rebuild
    south = fp.astar(np.array(SPAWN), goal)
    south_len = fp.path_length(np.array(SPAWN), goal)
    south_cross = _crosses_gate(south)
    fp.occupancy = occ_saved              # restore
    fp._occ_inflated = inf_saved
    print(f"[5] SOUTH detour (gate blocked)    : reachable={south is not None}  "
          f"gate-cells-on-path={south_cross}   "
          f"({'OK' if south is not None and (south_cross or 0)==0 else 'FAIL'})")
    print(f"[6] detour cost                    : west={west_len:.1f}m  south={south_len:.1f}m  "
          f"extra={south_len-west_len:+.1f}m ({100*(south_len-west_len)/max(west_len,1e-6):+.0f}%)")
    ok &= south is not None and (south_cross or 0) == 0

    # 7: no navigability regression — all ORIGINAL task goals reachable
    print("[7] original task goals reachable from spawn:")
    orig_tasks = {k: v for k, v in fp.tasks.items() if k != pe.PRAYER_TASK}
    for tid, tdef in orig_tasks.items():
        try:
            gc = fp.room_center(tdef["target_room_id"])
        except KeyError:
            continue
        g = fp.nearest_walkable(gc[0], gc[1], grid=fp.occupancy_inflated)
        p = fp.astar(np.array(SPAWN), g)
        flag = "OK" if p is not None else "FAIL(UNREACHABLE)"
        print(f"      {tid:14s} -> {tdef['target_room_id']:14s} {flag}")
        ok &= p is not None

    # engaging-task reachability too
    pg = fp.nearest_walkable(*pe.FW_WEST_CENTER, grid=fp.occupancy_inflated)
    p = fp.astar(np.array(SPAWN), pg)
    print(f"      {pe.PRAYER_TASK:14s} -> fw_west         "
          f"{'OK' if p is not None else 'FAIL(UNREACHABLE)'}")
    ok &= p is not None

    print("\n=== VALIDATION", "PASSED — safe to train ===" if ok else "FAILED — do NOT train ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
