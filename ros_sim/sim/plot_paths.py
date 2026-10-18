"""Publication figures from campaign runs: the robot's path over the floor plan.

    python3 plot_paths.py                              # every run in results_runs/
    python3 plot_paths.py --runs results_runs/walled__* --out results_figures
    python3 plot_paths.py --compare walled              # one panel per policy, one tier

Each figure draws the trained floor plan (walls, beds, nurse desk, the three
standing people), the cultural zones as a soft tint, the pedestrians' walking
loops (dashed, when the run had them), the patrol destinations, and the robot's
ground-truth trajectory coloured by commanded speed, with pedestrian-contact
events marked. Everything comes from the run's own CSV/JSON plus layout.py, so
a figure can always be regenerated from the files it is built from.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "hospitalnav_deploy", "hospitalnav_deploy"))
import layout as L                      # noqa: E402
import gen_worlds as G                  # noqa: E402
import ped_routes as P                  # noqa: E402
from zone_metrics import zone_metrics   # noqa: E402

PATH_RED = "#d7191c"          # the robot's travelled path (matches the RViz trail)
CONTACT = "#1a1a1a"
ZONE_TINT = {"male_ward": "#cfe0ff", "general_ward_1": "#d9f2e0", "female_ward": "#ffd9e8",
             "ward_icu_emerg": "#ffdcd6", "prayer": "#f5ecc8", "general_ward_3": "#dbe8f5",
             "general_ward_2": "#e4d9f5"}
CULT_HATCH = {"prayer": "//", "male_ward": "..", "female_ward": ".."}


def draw_floor(ax, tier, tr, peds_level="none"):
    ax.set_facecolor("#f4f4f2")
    for (x0, y0, x1, y1, name, _d) in L.ROOMS:
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=ZONE_TINT[name],
                                       edgecolor="none", hatch=CULT_HATCH.get(name), alpha=0.9, zorder=0))
        label = L.ROOM_LABEL[name] if x1 - x0 > 10 else L.ROOM_LABEL[name].replace(" ", "\n")
        ax.text((x0 + x1) / 2, y1 - 1.2 if y0 > 15 else y0 + 1.2, label,
                ha="center", va="center", fontsize=6, color="#444", zorder=3, linespacing=0.9)
    for p in tr["wall"]:
        idx = int(p.name.rsplit("_", 1)[1])
        if tier == "open" and idx >= 4:
            continue
        cx, cy = p.xy; sx, sy, _ = p.box
        ax.add_patch(patches.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, facecolor="#2b2b2b",
                                       edgecolor="none", zorder=2))
    for p in tr["bed"]:
        cx, cy = p.xy; sx, sy, _ = p.box
        ax.add_patch(patches.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, facecolor="#b9bfc7",
                                       edgecolor="#7d848c", lw=0.4, zorder=2))
    for p in tr.get("nurse_desk", []):
        cx, cy = p.xy; sx, sy, _ = p.box
        ax.add_patch(patches.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, facecolor="#a9825c",
                                       edgecolor="#6b4f33", lw=0.4, zorder=2))
    for p in tr["person"]:
        cx, cy = p.xy; r, _ = p.cyl
        ax.add_patch(patches.Circle((cx, cy), r, facecolor="#f2a541", edgecolor="#8a5a12", lw=0.5, zorder=3))
    if peds_level and peds_level != "none":
        names = P.LEVELS.get(peds_level, [])
        dense = len(names) > 10                       # crowd45: 45 loops, drawn faintly
        for n in names:
            pts = P.ROUTES[n][2] + [P.ROUTES[n][2][0]]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], ls=(0, (3, 2)), lw=0.4 if dense else 0.7,
                    color="#d9822b", alpha=0.35 if dense else 0.75, zorder=3)
    ax.set_xlim(-0.5, L.MAP_W + 0.5); ax.set_ylim(-0.5, L.MAP_H + 0.5)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.tick_params(labelsize=7)


def load_run(d):
    rec = json.load(open(os.path.join(d, "result.json"))) if os.path.exists(os.path.join(d, "result.json")) else {}
    rows = list(csv.DictReader(open(os.path.join(d, "run.csv"))))
    return rec, rows


def tier_of(tag):
    return tag.split("__")[0]


def draw_run(ax, rec, rows, tr, tier, peds_level, title=None, by_speed=False):
    draw_floor(ax, tier, tr, peds_level)
    rows = [r for r in rows if r.get("goal_x", "nan") != "nan"] or rows     # from the first goal on
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    v = np.array([abs(float(r["v_cmd"])) for r in rows])
    if len(xy) > 1:
        if by_speed:
            segs = np.stack([xy[:-1], xy[1:]], axis=1)
            lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, 0.8), lw=1.8, zorder=5)
            lc.set_array(v[:-1])
            ax.add_collection(lc)
        else:
            ax.plot(xy[:, 0], xy[:, 1], color=PATH_RED, lw=1.9, solid_capstyle="round", zorder=5,
                    label="robot path")
    ax.plot(xy[0, 0], xy[0, 1], ls="none", marker="o", ms=6, color="#2ca02c", mec="k", zorder=6, label="start")
    for g in rec.get("goals", []):
        ok = g.get("outcome") == "reached"
        ax.plot(g["x"], g["y"], ls="none", marker="*" if ok else "X", ms=11 if ok else 8,
                color="#1f5fbf" if ok else "#111111", mec="k", mew=0.5, zorder=7,
                label="goal (reached)" if ok else "goal (timeout)")
    # pedestrian contacts: ground-truth centre distance below 0.5 m (robot + body radius)
    if rows and "ped_min_m" in rows[0]:
        pm = np.array([float(r["ped_min_m"]) if r["ped_min_m"] != "nan" else np.nan for r in rows])
        hit = np.where(pm < 0.5)[0]
        if hit.size:
            ax.plot(xy[hit, 0], xy[hit, 1], ls="none", marker="D", ms=4.5, mfc="none", mec=CONTACT,
                    mew=1.0, zorder=8, label="pedestrian within 0.5 m")
    if title:
        ax.set_title(title, fontsize=9)


def legend(fig, ax, peds=True, fontsize=8.5, y=-0.01):
    """One-row legend under the panels (short labels so it fits even the
    single-run figure)."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=PATH_RED, lw=2.2, label="robot path"),
         Line2D([], [], ls="none", marker="o", ms=6, color="#2ca02c", mec="k", label="start (nurse desk)"),
         Line2D([], [], ls="none", marker="*", ms=11, color="#1f5fbf", mec="k", label="goal reached"),
         Line2D([], [], ls="none", marker="X", ms=8, color="#111111", mec="k", label="not reached"),
         patches.Patch(facecolor=ZONE_TINT["prayer"], edgecolor="#555", hatch="//", label="prayer room"),
         patches.Patch(facecolor=ZONE_TINT["male_ward"], edgecolor="#555", hatch="..", label="gendered ward")]
    if peds:
        h += [Line2D([], [], ls=(0, (3, 2)), color="#d9822b", lw=1.0, label="pedestrian loops"),
              Line2D([], [], ls="none", marker="D", ms=5, mfc="none", mec=CONTACT, label="pedestrian < 0.5 m")]
    return fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=fontsize, frameon=False,
                      bbox_to_anchor=(0.5, y), handlelength=1.5, handletextpad=0.5, columnspacing=1.2)


def save(fig, path_png):
    fig.savefig(path_png, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path_png[:-4] + ".pdf", bbox_inches="tight", pad_inches=0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run directories (default: all in results_runs)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_figures"))
    ap.add_argument("--compare", default="", help="tier name: one panel per policy (escape on, same peds)")
    ap.add_argument("--peds", default="busy", help="pedestrian level for --compare panels")
    ap.add_argument("--by-speed", action="store_true", help="colour the path by commanded speed instead of red")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = G.load_trained()
    runs = a.runs or sorted(d for d in glob.glob(os.path.join(ROOT, "results_runs", "*")) if os.path.isdir(d))
    runs = [d for d in runs if os.path.exists(os.path.join(d, "run.csv"))]
    if a.compare:
        sel = [d for d in runs if os.path.basename(d).startswith(a.compare + "__") and "escape-on" in d
               and os.path.basename(d).endswith(f"peds-{a.peds}")]
        n = len(sel)
        if not n:
            raise SystemExit("no runs match")
        cols = min(3, n); rows_ = math.ceil(n / cols)
        # explicit layout (inches): every map panel has the floor's own aspect,
        # tick labels only on the outer edges, the legend right under the panels
        pw = 4.0
        ph = pw * (L.MAP_H + 1.0) / (L.MAP_W + 1.0)
        title_h, left, right, top, gap_c, gap_r, xt_h, leg_h = 0.40, 0.42, 0.06, 0.02, 0.10, 0.06, 0.40, 0.38
        W = left + cols * pw + (cols - 1) * gap_c + right
        H = top + rows_ * (title_h + ph) + (rows_ - 1) * gap_r + xt_h + leg_h
        fig = plt.figure(figsize=(W, H))
        axl = []
        for i in range(rows_ * cols):
            r, c = divmod(i, cols)
            x0 = left + c * (pw + gap_c)
            y0 = H - top - r * (title_h + ph + gap_r) - title_h - ph
            axl.append(fig.add_axes([x0 / W, y0 / H, pw / W, ph / H]))
        axs = np.array(axl).reshape(rows_, cols)
        for ax, d in zip(axs.flat, sel):
            rec, rows = load_run(d)
            tag = os.path.basename(d)
            goals = rec.get("goals", []); nr = sum(1 for g in goals if g.get("outcome") == "reached")
            ped = rec.get("pedestrians") or {}
            zm = zone_metrics(os.path.join(d, "run.csv"))["prayer"]
            draw_run(ax, rec, rows, tr, a.compare, rec.get("peds", tag.split("peds-")[-1] if "peds-" in tag else "none"),
                     by_speed=a.by_speed,
                     title=f"$\\bf{{{tag.split('__')[1].replace('_', chr(92) + '_')}}}$: {rec.get('outcome','?')}, "
                           f"{rec.get('path_length_m','-')} m\nprayer depth {zm['max_depth_m']:.2f} m, "
                           f"{ped.get('contacts','-')} ped contacts")
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
        for i, ax in enumerate(axs.flat):
            r, c = divmod(i, cols)
            if c > 0:
                ax.set_ylabel(""); ax.tick_params(labelleft=False)
            below = i + cols < n                          # a panel sits under this one
            if below:
                ax.set_xlabel(""); ax.tick_params(labelbottom=False)
        if a.by_speed:
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 0.8))
            fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.6, label="commanded speed (m/s)")
        else:
            legend(fig, axs.flat[0], peds=a.peds != "none", fontsize=8.5, y=0.004 + 0.02 / H)
        out = os.path.join(a.out, f"paths_{a.compare}_peds-{a.peds}_compare.png")
        save(fig, out); plt.close(fig)
        print("wrote", out)
        return 0
    for d in runs:
        rec, rows = load_run(d)
        if not rows:
            continue
        tag = os.path.basename(d)
        tier = tier_of(tag)
        # result.json does not carry the pedestrian level; the run directory's name does
        tag_ = os.path.basename(d.rstrip("/"))
        peds_level = rec.get("peds") or (tag_.split("peds-")[-1].split("__")[0] if "peds-" in tag_ else "none")
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        goals = rec.get("goals", []); nr = sum(1 for g in goals if g.get("outcome") == "reached")
        ped = rec.get("pedestrians") or {}
        zm = zone_metrics(os.path.join(d, "run.csv"))["prayer"]
        draw_run(ax, rec, rows, tr, tier, peds_level, by_speed=a.by_speed,
                 title=f"{tag}\n{rec.get('outcome','?')}: {nr}/{len(goals)} goals, path {rec.get('path_length_m','-')} m, "
                       f"sim {rec.get('sim_time_s','-')} s, escapes {rec.get('escapes',{}).get('stall','-')}/"
                       f"{rec.get('escapes',{}).get('wedge','-')}, ped contacts {ped.get('contacts','-')} "
                       f"(min {ped.get('min_distance_m','-')} m), prayer depth {zm['max_depth_m']:.2f} m")
        if a.by_speed:
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 0.8))
            fig.colorbar(sm, ax=ax, shrink=0.75, label="commanded speed (m/s)")
        else:
            legend(fig, ax, peds=peds_level != "none", fontsize=5.6)
        out = os.path.join(a.out, f"path_{tag}.png")
        save(fig, out); plt.close(fig)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
