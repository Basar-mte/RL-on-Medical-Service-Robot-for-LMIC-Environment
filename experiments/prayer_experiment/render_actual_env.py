"""Render the ACTUAL generated env (from dmch_redesign.yaml via FloorPlan) so we
confirm the navigable env matches the concept before training. Overlays the A*
path from spawn to each task goal to show routes are real."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from envs.floorplan import FloorPlan, CODE_TO_ZONE  # noqa: E402
from figures.make_figures import ZONE_RGB  # noqa: E402

YAML = Path(__file__).resolve().parent / "dmch_redesign.yaml"


def main():
    fp = FloorPlan(str(YAML))
    H, W = fp.height_cells, fp.width_cells
    rgb = np.ones((H, W, 3))
    for r in range(H):
        for c in range(W):
            if fp.occupancy[r, c]:
                rgb[r, c] = ZONE_RGB["wall"]
            else:
                rgb[r, c] = ZONE_RGB.get(CODE_TO_ZONE.get(int(fp.zone_map[r, c]), "walkable"), ZONE_RGB["walkable"])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(rgb, origin="lower", extent=[0, fp.width_m, 0, fp.height_m], interpolation="nearest")
    spawn = np.array([16.5, 10.4])
    ax.scatter(*spawn, s=140, marker="o", facecolor="white", edgecolor="black", lw=2, zorder=6)
    cols = plt.cm.tab10(np.linspace(0, 1, len(fp.tasks)))
    for (tid, t), col in zip(fp.tasks.items(), cols):
        gc = fp.room_center(t["target_room_id"])
        g = fp.nearest_walkable(gc[0], gc[1], grid=fp.occupancy_inflated)
        path = fp.astar(spawn, g)
        if path:
            xs = [p[0] for p in path]; ys = [p[1] for p in path]
            ax.plot(xs, ys, color=col, lw=2, alpha=0.9, label=f"{tid}→{t['target_room_id']}", zorder=5)
            ax.scatter(g[0], g[1], s=70, marker="*", color=col, edgecolor="black", lw=0.6, zorder=6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.set_xlim(0, 40); ax.set_ylim(0, 30); ax.set_aspect("equal")
    ax.set_title("ACTUAL navigable env (dmch_redesign.yaml) + A* task routes from spawn", fontsize=12)
    fig.tight_layout()
    out = ROOT / "experiments" / "prayer_experiment" / "env_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
