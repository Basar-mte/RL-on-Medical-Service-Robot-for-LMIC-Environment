"""Paper figure: the prayer-shortcut routes on Tier 2 under three planner settings
(geometric; cultural inside a prayer window; cultural outside), single row, vector PDF.
    .venv\\Scripts\\python.exe scripts\\make_routes_figure.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.floorplan import load_floorplan  # noqa: E402
from planner.planner import Planner  # noqa: E402

fp = load_floorplan(ROOT / "configs" / "hospital_v4.yaml")
TASK = "STAT_medicine"
goal = fp.nearest_walkable(*fp.room_center(fp.tasks[TASK]["target_room_id"]), grid=fp.occupancy_inflated)
VARIANTS = [("geometric ($\\kappa_{\\mathrm{c}}{=}0$)", 0.0, 12.5),
            ("cultural, prayer window", 1.0, 12.5),
            ("cultural, outside window", 1.0, 10.0)]

bg = np.ones(fp.zone_map.shape + (3,))
for code, col in {0: (.35, .35, .35), 8: (.62, .85, .62), 9: (.68, .8, .92), 10: (.93, .85, .74),
                  2: (.9, .9, .78), 3: (.8, .92, .8)}.items():
    bg[fp.zone_map == code] = col
bg[fp.occupancy.astype(bool)] = (.35, .35, .35)

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05))
for ax, (name, ratio, tod) in zip(axes, VARIANTS):
    ax.imshow(bg, origin="lower", extent=(0, fp.width_m, 0, fp.height_m))
    f = Planner(fp, ratio=ratio).field(goal, tod)
    for sp in fp.robot_spawn_points:
        sp = fp.nearest_walkable(sp[0], sp[1], grid=fp.occupancy_inflated)
        if np.linalg.norm(sp - goal) < 3.0:
            continue
        route = f.route_xy(sp)
        cells = [fp._world_to_cell(x, y) for x, y in route]
        in_prayer = any(fp.zone_map[r, c] == 8 for r, c in cells)
        ax.plot(route[:, 0], route[:, 1], lw=1.7, color=("#c62828" if in_prayer else "#1565c0"),
                solid_capstyle="round")
        ax.plot(sp[0], sp[1], "o", color="black", ms=3.2)
    ax.plot(goal[0], goal[1], "*", color="#f9a825", ms=13, mec="black", mew=0.7)
    ax.text(13, 7.5, "prayer", ha="center", va="center", fontsize=6.6, color="#1b5e20")
    ax.set_title(name, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 40); ax.set_ylim(0, 17.5)   # south half: corridor + rooms (the routes' extent)
plt.tight_layout(pad=0.3)
out = ROOT / "manuscript" / "figures" / "fig10_planner_routes.pdf"
plt.savefig(out); plt.savefig(out.with_suffix(".png"), dpi=220)
print("wrote", out)
