"""Validate the generated DMCH-redesign YAML before any training.

Checks (all on the radius-inflated robot grid):
  1. every robot spawn point is robot-walkable
  2. every TASK-TARGET room center is reachable from the primary spawn (A*)
  3. the mosque (prayer) is reachable (prayer_service task)
  4. reports the walkable-area fraction and any unreachable rooms
Exit 0 only if all task goals are reachable.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from envs.floorplan import FloorPlan  # noqa: E402

YAML = Path(__file__).resolve().parent / "dmch_redesign.yaml"


def main():
    fp = FloorPlan(str(YAML))
    inf = fp.occupancy_inflated
    walk_frac = 1.0 - inf.mean()
    print(f"map {fp.width_m}x{fp.height_m}  walkable(robot) fraction = {walk_frac:.2%}")

    ok = True
    print("\n[1] robot spawn points walkable(robot):")
    spawns = fp.robot_spawn_points
    for s in spawns:
        w = fp.is_walkable_robot(s[0], s[1])
        print(f"    {s} -> {'OK' if w else 'BLOCKED'}")
    # primary spawn: first walkable one (snap if needed)
    prim = None
    for s in spawns:
        if fp.is_walkable_robot(s[0], s[1]):
            prim = np.array(s, dtype=float); break
    if prim is None:
        prim = fp.nearest_walkable(spawns[0][0], spawns[0][1], grid=fp.occupancy_inflated)
    print(f"    primary spawn = {tuple(np.round(prim,2))}")

    print("\n[2] task-goal reachability from primary spawn (A*):")
    targets = {tid: t["target_room_id"] for tid, t in fp.tasks.items()}
    for tid, room in targets.items():
        try:
            gc = fp.room_center(room)
        except KeyError:
            print(f"    {tid:14s} -> room '{room}' MISSING"); ok = False; continue
        g = fp.nearest_walkable(gc[0], gc[1], grid=fp.occupancy_inflated)
        path = fp.astar(prim, g)
        length = fp.path_length(prim, g) if path else -1
        flag = "OK" if path is not None else "UNREACHABLE"
        if path is None:
            ok = False
        print(f"    {tid:14s} -> {room:14s} {flag:12s} ({length:.1f} m, goal {tuple(np.round(g,1))})")

    print("\n[3] cultural-zone sanity:")
    for name, pt in [("mosque(prayer)", (12, 27)), ("male_ward", (4.5, 12)), ("female_ward", (35.5, 12))]:
        print(f"    zone at {pt} = {fp.get_zone(*pt)!r}")
    pc = fp.nearest_room_of_type(20, 14, "prayer")
    print(f"    nearest prayer room center = {None if pc is None else tuple(np.round(pc,1))}")

    print("\n=== VALIDATION", "PASSED — safe to train ===" if ok else "FAILED — fix before training ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
