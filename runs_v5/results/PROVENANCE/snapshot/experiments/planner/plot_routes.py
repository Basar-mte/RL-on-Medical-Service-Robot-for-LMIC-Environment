"""Draw the planner's routes for every task from every spawn point (geometric vs
cultural planner, in and out of a prayer window) and print their lengths and
prayer-room transit. Sanity check of the planner + a paper figure candidate.

    python experiments/planner/plot_routes.py [configs/hospital_v4.yaml] [out.png]
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from envs.floorplan import load_floorplan, ZONE_CODES  # noqa: E402
from envs.lmic_hospital_env import TASK_IDS  # noqa: E402
from planner.planner import Planner  # noqa: E402

cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/hospital_v4.yaml"
out = sys.argv[2] if len(sys.argv) > 2 else f"experiments/planner/routes_{Path(cfg).stem}.png"
fp = load_floorplan(ROOT / cfg)

VARIANTS = [("geometric", 0.0, 12.5), ("cultural, prayer window", 1.0, 12.5), ("cultural, outside window", 1.0, 10.0)]
planners = {name: Planner(fp, ratio=r) for name, r, _ in VARIANTS}
prayer = fp.zone_map == ZONE_CODES["prayer"]

fig, axes = plt.subplots(len(TASK_IDS), len(VARIANTS), figsize=(4.2 * len(VARIANTS), 3.3 * len(TASK_IDS)))
bg = np.ones(fp.zone_map.shape + (3,))
for code, col in {0: (.3, .3, .3), 8: (.5, .8, .5), 9: (.6, .75, .9), 10: (.9, .8, .7), 2: (.85, .85, .7)}.items():
    bg[fp.zone_map == code] = col
bg[fp.occupancy.astype(bool)] = (.3, .3, .3)

t0 = time.time()
for i, task in enumerate(TASK_IDS):
    room = fp.tasks[task]["target_room_id"]
    goal = fp.nearest_walkable(*fp.room_center(room), grid=fp.occupancy_inflated)
    for j, (name, ratio, tod) in enumerate(VARIANTS):
        ax = axes[i, j]
        ax.imshow(bg, origin="lower", extent=(0, fp.width_m, 0, fp.height_m))
        f = planners[name].field(goal, tod)
        for sp in fp.robot_spawn_points:
            sp = fp.nearest_walkable(sp[0], sp[1], grid=fp.occupancy_inflated)
            if np.linalg.norm(sp - goal) < 3.0:
                continue
            route = f.route_xy(sp)
            if len(route) == 0:
                print(f"  {task:<14} {name:<26} spawn {sp}: NO ROUTE"); continue
            cells = [fp._world_to_cell(x, y) for x, y in route]
            in_prayer = sum(prayer[r, c] for r, c in cells) * fp.cell_size
            L = f.geodesic(sp)
            print(f"  {task:<14} {name:<26} spawn ({sp[0]:4.1f},{sp[1]:4.1f}) route {L:5.1f} m, "
                  f"{in_prayer:4.1f} m inside prayer room")
            ax.plot(route[:, 0], route[:, 1], lw=1.6, color="red" if in_prayer > 0 else "blue")
            ax.plot(sp[0], sp[1], "ko", ms=3)
        ax.plot(goal[0], goal[1], "y*", ms=11, mec="k")
        ax.set_title(f"{task} — {name}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(ROOT / out, dpi=110)
print(f"saved {out}  ({time.time() - t0:.1f} s incl. field computation)")
