"""Render the prayer-ENGAGED benchmark variant used by the probe.

Shows the relocated prayer zone (now a "gate" on the female-ward west-door
approach), the cleared former corner room, and the two routes the engaging
prayer_call task can take: the SHORT west route THROUGH the prayer gate vs the
prayer-avoiding SOUTH detour. Reuses the paper's fig1 palette for consistency.

Writes prayer_experiment/prayer_env.png and .pdf. Reads only the engaged env.
Run:  .venv\\Scripts\\python.exe prayer_experiment\\make_env_figure.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "prayer_experiment"))

import prayer_engage as pe            # noqa: E402  (engaged env)
import envs.lmic_hospital_env as em   # noqa: E402
from figures.make_figures import (    # noqa: E402
    _render_fp, ZONE_RGB, BED_COLOR, WARD_BED_COLOR, MEJHE_COLOR,
    EQUIP_COLOR, BENCH_COLOR)

CONFIG = ROOT / "configs" / "dmch_hospital.yaml"
SPAWN = np.array([20.0, 13.5])


def _plen(path):
    if not path or len(path) < 2:
        return 0.0
    return sum(float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1))


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fp = em.load_floorplan(str(CONFIG))              # engaged loader
    rgb, beds, ward_beds, mejhe, equip, benches = _render_fp(fp, cfg)

    goal = fp.nearest_walkable(*pe.FW_WEST_CENTER, grid=fp.occupancy_inflated)

    # route A: unconstrained shortest path (west, through the gate)
    west = fp.astar(SPAWN, goal)
    # route B: south detour — temporarily wall the gate, replan
    occ, inf = fp.occupancy.copy(), fp._occ_inflated
    cs = fp.cell_size
    c0, c1 = int(pe.GATE_X0 / cs), int(np.ceil(pe.GATE_X1 / cs))
    r0, r1 = int(pe.GATE_Y0 / cs), int(np.ceil(pe.GATE_Y1 / cs))
    fp.occupancy[r0:r1, c0:c1] = 1
    fp._occ_inflated = None
    south = fp.astar(SPAWN, goal)
    fp.occupancy, fp._occ_inflated = occ, inf
    wl, sl = _plen(west), _plen(south)

    fig, ax = plt.subplots(figsize=(12, 8.4))
    ax.imshow(rgb, origin="lower", extent=[0, fp.width_m, 0, fp.height_m],
              interpolation="nearest", zorder=0)

    def draw(rects, color):
        for (x0, y0, w, h) in rects:
            ax.add_patch(mpatches.Rectangle((x0, y0), w, h, facecolor=color,
                                            edgecolor="none", zorder=1))
    draw(beds, BED_COLOR); draw(ward_beds, WARD_BED_COLOR)
    draw(mejhe, MEJHE_COLOR); draw(equip, EQUIP_COLOR); draw(benches, BENCH_COLOR)

    # ---- prayer gate highlight + callout -------------------------------------
    ax.add_patch(mpatches.Rectangle((pe.GATE_X0, pe.GATE_Y0),
                                    pe.GATE_X1 - pe.GATE_X0, pe.GATE_Y1 - pe.GATE_Y0,
                                    facecolor="none", edgecolor="#1b7837", lw=2.4,
                                    hatch="////", zorder=6))
    ax.annotate("PRAYER GATE\n(relocated here)", xy=(21.0, 18.4), xytext=(9.5, 25.2),
                fontsize=10, fontweight="bold", color="#14532d", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#14532d", lw=1.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="#eaf6ea", ec="#1b7837", lw=1.2),
                zorder=8)
    ax.annotate("former prayer room\n→ cleared to walkable", xy=(33.0, 26.0),
                xytext=(30.0, 27.4), fontsize=9, style="italic", color="#444",
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.2), zorder=8)

    # ---- routes --------------------------------------------------------------
    def route(path, color, label, ls="-"):
        if path is None:
            return
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, color=color, lw=3.0, ls=ls, label=label,
                solid_capstyle="round", zorder=7)
    route(west, "#b2182b", f"West route — through prayer gate ({wl:.1f} m)")
    route(south, "#2166ac", f"South detour — prayer-avoiding ({sl:.1f} m, +{100*(sl-wl)/wl:.0f}%)", ls=(0, (5, 2)))

    # ---- doors + spawn + goal ------------------------------------------------
    ax.annotate("female-ward\nWEST door", xy=(22.0, 18.7), xytext=(15.6, 20.9),
                fontsize=8.5, color="#7a1f1f", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#7a1f1f", lw=1.2), zorder=8)
    ax.annotate("female-ward\nSOUTH door", xy=(31.2, 15.0), xytext=(35.0, 12.3),
                fontsize=8.5, color="#1a4a7a", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#1a4a7a", lw=1.2), zorder=8)

    ax.scatter(*SPAWN, s=180, marker="o", facecolor="white", edgecolor="black",
               lw=2.0, zorder=9)
    ax.text(SPAWN[0], SPAWN[1] - 1.2, "Spawn\n(nurse station)", fontsize=8.5,
            ha="center", va="top", fontweight="bold", zorder=9)
    ax.scatter(goal[0], goal[1], s=220, marker="*", facecolor="#f0c000",
               edgecolor="black", lw=1.6, zorder=9)
    ax.text(goal[0] + 0.6, goal[1] + 0.9, "Goal\n(prayer_call)", fontsize=8.5,
            ha="left", va="bottom", fontweight="bold", zorder=9)

    # ---- legend --------------------------------------------------------------
    zone_items = [
        ("prayer", "Prayer zone (gate)"), ("female_ward", "Female ward"),
        ("male_ward", "Male ward"), ("nurse_station", "Nurse station"),
        ("emergency", "Emergency"), ("icu", "ICU"),
        ("pharmacy", "Pharmacy / Lab"), ("ward", "General ward"),
        ("corridor", "Corridor"), ("wall", "Wall / obstacle"),
    ]
    handles = [mpatches.Patch(facecolor=ZONE_RGB[z], edgecolor="#888", lw=0.4, label=lab)
               for z, lab in zone_items]
    handles += [
        Line2D([0], [0], color="#b2182b", lw=3, label="West route (through gate)"),
        Line2D([0], [0], color="#2166ac", lw=3, ls=(0, (5, 2)), label="South detour (avoids gate)"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, frameon=True, borderaxespad=0.0, title="Legend")

    ax.set_xlim(0, fp.width_m); ax.set_ylim(0, fp.height_m)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("LMIC-HospitalNav-v1 — prayer-engaged probe variant\n"
                 "prayer avoidance made a live, time-varying constraint (west door gated; "
                 "south door is the +68% compliant detour)", fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = ROOT / "prayer_experiment" / f"prayer_env.{ext}"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
