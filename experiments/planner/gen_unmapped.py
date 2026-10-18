"""Generate 'unmapped obstacle' world variants of a floor-plan config: K box obstacles
(0.6-1.5 m) dropped on walkable space, unknown to the route planner (which keeps using
the original config via PLANNER_MAP). Layouts are seeded, keep clear of robot spawns
(>= 1.8 m) and task-goal centres (>= 1.2 m), and are accepted only if every task goal
stays reachable from every spawn on the modified map.

    python experiments/planner/gen_unmapped.py [configs/hospital_v4.yaml] [n_layouts] [K]
Outputs: experiments/planner/unmapped_cfgs/<stem>_L<i>.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from envs.floorplan import FloorPlan  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "configs/hospital_v4.yaml"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
OUT = Path(__file__).resolve().parent / "unmapped_cfgs"
OUT.mkdir(exist_ok=True)

cfg0 = yaml.safe_load((ROOT / BASE).read_text(encoding="utf-8"))
fp0 = FloorPlan(ROOT / BASE)
spawns = [np.array(p, float) for p in fp0.robot_spawn_points]
goals = []
for t, td in fp0.tasks.items():
    try:
        goals.append((t, fp0.nearest_walkable(*fp0.room_center(td["target_room_id"]), grid=fp0.occupancy_inflated)))
    except KeyError:
        pass


def reachable(fp):
    return all(fp.astar(sp, g) is not None for sp in spawns for _, g in goals)


made = 0
for i in range(N):
    rng = np.random.default_rng(7000 + i)
    for attempt in range(60):
        boxes = []
        while len(boxes) < K:
            x, y = rng.uniform(1, fp0.width_m - 1), rng.uniform(1, fp0.height_m - 1)
            w, h = rng.uniform(0.6, 1.5), rng.uniform(0.6, 1.5)
            c = np.array([x, y])
            if not fp0.is_walkable_robot(x, y):
                continue
            if any(np.linalg.norm(c - sp) < 1.8 for sp in spawns):
                continue
            if any(np.linalg.norm(c - g) < 1.2 for _, g in goals):
                continue
            boxes.append((x, y, w, h))
        cfg = yaml.safe_load((ROOT / BASE).read_text(encoding="utf-8"))
        for j, (x, y, w, h) in enumerate(boxes):
            cfg["static_obstacles"].append(dict(
                polygon=[[round(x - w / 2, 2), round(y - h / 2, 2)], [round(x + w / 2, 2), round(y - h / 2, 2)],
                         [round(x + w / 2, 2), round(y + h / 2, 2)], [round(x - w / 2, 2), round(y + h / 2, 2)]],
                label=f"Unmapped box {j}"))
        out = OUT / f"{Path(BASE).stem}_L{i}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        if reachable(FloorPlan(out)):
            print(f"layout {i}: {K} boxes, attempt {attempt + 1}, all {len(goals)} goals reachable -> {out.name}")
            made += 1
            break
        out.unlink()
    else:
        print(f"layout {i}: FAILED to find a reachable layout")
print(f"{made}/{N} layouts written to {OUT}")
