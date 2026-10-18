"""DMCH redesign v4 — PORTRAIT 30 (wide) x 40 (tall). CONCEPT schematic (DRAFT).

Same room logic as v3, re-laid in a 30x40 portrait footprint (entrance at the
bottom, exit at the top, long gender wards down the sides, central general ward):
  * Male / Female wards down the left / right, dense, gates kept CLEAR of beds.
  * Central GENERAL WARD island (4 doors) with the NURSE STATION in its corner
    and beds lined around its perimeter.
  * Mosque (Namaz Ghor, male-only) + Wudu and ICU at the top; intern doctor room
    + staff room per ward; emergency + Ansar + trolley, pharmacy, store room at
    the bottom; vertical corridor-overflow beds against the ward walls; family
    attendants throughout. No radiology / storage-soiled / ticket registry.

Pure matplotlib. Writes prayer_experiment/redesign_bd4.png/.pdf.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent

WALL   = np.array([0.25, 0.26, 0.30]); CORR = np.array([0.90, 0.91, 0.93])
MALEBG = np.array([0.83, 0.91, 0.98]); MALEBED = np.array([0.30, 0.52, 0.74])
FEMBG  = np.array([0.99, 0.90, 0.90]); FEMBED  = np.array([0.83, 0.47, 0.55])
GENBG  = np.array([0.94, 0.90, 0.72]); GENBED  = np.array([0.80, 0.63, 0.30])
MOSQUE = np.array([0.56, 0.83, 0.50]); WUDU = np.array([0.66, 0.86, 0.88])
OVER   = np.array([0.96, 0.69, 0.26]); ICUC = np.array([0.78, 0.55, 0.85]); ICUBED = np.array([0.55, 0.30, 0.62])
MEJHE  = np.array([0.93, 0.90, 0.48]); SERV = np.array([0.80, 0.80, 0.84])
NURSE  = np.array([0.40, 0.76, 0.73])
EMERG  = np.array([0.95, 0.61, 0.57]); PHARM = np.array([0.60, 0.67, 0.86])
INTERN = np.array([0.72, 0.83, 0.66]); STAFFR = np.array([0.86, 0.80, 0.66])
ATTEND = "#6a51a3"


def _sub(lo, hi, gaps):
    segs, cur = [], lo
    for a, b in sorted(gaps):
        a, b = max(a, lo), min(b, hi)
        if a > cur: segs.append((cur, a))
        cur = max(cur, b)
    if cur < hi: segs.append((cur, hi))
    return segs


def walls(ax, x0, y0, x1, y1, gates=(), lw=3.0):
    for side, e in (("S", y0), ("N", y1)):
        for a, b in _sub(x0, x1, [(a, b) for s, a, b in gates if s == side]):
            ax.plot([a, b], [e, e], color=WALL, lw=lw, solid_capstyle="butt", zorder=6)
    for side, e in (("W", x0), ("E", x1)):
        for a, b in _sub(y0, y1, [(a, b) for s, a, b in gates if s == side]):
            ax.plot([e, e], [a, b], color=WALL, lw=lw, solid_capstyle="butt", zorder=6)


def fill(ax, x0, y0, x1, y1, c, z=1):
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=c, edgecolor="none", zorder=z))


def bed(ax, x, y, w, h, c, head="left"):
    ax.add_patch(mpatches.FancyBboxPatch((x + 0.05, y + 0.05), w - 0.1, h - 0.1,
                 boxstyle="round,pad=0,rounding_size=0.14", facecolor=c, edgecolor=WALL, lw=0.5, zorder=4))
    pw = 0.32
    if head in ("left", "right"):
        px = x + 0.1 if head == "left" else x + w - pw - 0.1
        ax.add_patch(mpatches.Rectangle((px, y + 0.16), pw, h - 0.32, facecolor="white", edgecolor=WALL, lw=0.4, zorder=5))
    else:
        py = y + 0.1 if head == "bottom" else y + h - pw - 0.1
        ax.add_patch(mpatches.Rectangle((x + 0.16, py), w - 0.32, pw, facecolor="white", edgecolor=WALL, lw=0.4, zorder=5))


def bed_col_skip(ax, x, y0, y1, pitch, w, h, c, head, skips=()):
    y = y0
    while y + h <= y1 + 1e-6:
        if all(y + h <= s0 or y >= s1 for (s0, s1) in skips):
            bed(ax, x, y, w, h, c, head)
        y += pitch


def mejhe(ax, x, y, w=1.2, h=0.55):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=MEJHE, edgecolor=WALL, lw=0.3, zorder=4))


def person(ax, x, y, c=ATTEND, s=0.75, z=8, a=0.9):
    ax.add_patch(mpatches.Circle((x, y + 0.24 * s), 0.14 * s, facecolor=c, edgecolor="none", zorder=z, alpha=a))
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.14 * s, y - 0.15 * s), 0.28 * s, 0.37 * s,
                 boxstyle="round,pad=0,rounding_size=0.08", facecolor=c, edgecolor="none", zorder=z, alpha=a))


_OFF = [(-0.28, 0.0), (0.28, 0.05), (0.0, -0.26), (-0.24, -0.28), (0.26, -0.32)]


def cluster(ax, cx, cy, n, s=0.72, c=ATTEND):
    for i in range(min(n, len(_OFF))):
        dx, dy = _OFF[i]
        person(ax, cx + dx, cy + dy, c=c, s=s)


def side_ward(ax, x0, x1, bg, bedc, headL_x, headR_x, gate_side, over_x, label):
    """Draw a tall side ward (male/female) y in [5,31] with 2 bed columns,
    2 clear gates, vertical overflow beds outside the gate wall, attendants."""
    MG = [(12.0, 14.0), (23.0, 25.0)]
    fill(ax, x0, 5, x1, 31, bg)
    bed_col_skip(ax, headL_x, 5.3, 30.6, 2.55, 2.0, 1.65, bedc, "left", skips=(MG if gate_side == "W" else ()))
    bed_col_skip(ax, headR_x, 5.3, 30.6, 2.55, 2.0, 1.65, bedc, "right", skips=(MG if gate_side == "E" else ()))
    for i in range(10):
        yc = 6.0 + i * 2.55
        if all(yc <= s0 or yc >= s1 for s0, s1 in MG) and yc < 30:
            mejhe(ax, (x0 + x1) / 2 - 0.6, yc - 0.2, 1.2, 0.5)
            cluster(ax, (x0 + x1) / 2, yc, 2)
    g = [(gate_side, 12.0, 14.0), (gate_side, 23.0, 25.0)]
    walls(ax, x0, 5, x1, 31, gates=g)
    bed_col_skip(ax, over_x, 5.4, 30.6, 2.55, 1.45, 2.1, OVER, "top", skips=MG)
    ax.text((x0 + x1) / 2, 30.2, label, ha="center", fontsize=8.2, fontweight="bold", zorder=9)


def main():
    fig, ax = plt.subplots(figsize=(10.5, 13.2))
    fill(ax, 0, 0, 30, 40, CORR, z=0)

    # ================= SIDE WARDS ============================================
    side_ward(ax, 0, 8, MALEBG, MALEBED, 0.35, 5.65, "E", 8.15, "MALE WARD (Purush)")
    side_ward(ax, 22, 30, FEMBG, FEMBED, 22.35, 27.65, "W", 20.4, "FEMALE WARD (Mohila)")

    # intern doctor room + staff room per ward (top corners)
    fill(ax, 0, 31, 4, 40, INTERN, z=2); walls(ax, 0, 31, 4, 40, gates=[("S", 1.3, 2.7)])
    ax.text(2, 35.5, "INTERN\nDOCTOR\nROOM", ha="center", va="center", fontsize=6.6, fontweight="bold", zorder=9)
    fill(ax, 4, 31, 8, 40, STAFFR, z=2); walls(ax, 4, 31, 8, 40, gates=[("S", 5.3, 6.7)])
    ax.text(6, 35.5, "STAFF\nROOM", ha="center", va="center", fontsize=6.6, fontweight="bold", zorder=9)
    fill(ax, 22, 31, 26, 40, INTERN, z=2); walls(ax, 22, 31, 26, 40, gates=[("S", 23.3, 24.7)])
    ax.text(24, 35.5, "INTERN\nDOCTOR\nROOM", ha="center", va="center", fontsize=6.6, fontweight="bold", zorder=9)
    fill(ax, 26, 31, 30, 40, STAFFR, z=2); walls(ax, 26, 31, 30, 40, gates=[("S", 27.3, 28.7)])
    ax.text(28, 35.5, "STAFF\nROOM", ha="center", va="center", fontsize=6.6, fontweight="bold", zorder=9)

    # ================= CENTRE: GENERAL WARD (4 doors) + nurse corner ==========
    gx0, gy0, gx1, gy1 = 11, 12, 19, 26
    fill(ax, gx0, gy0, gx1, gy1, GENBG, z=2)
    fill(ax, 11, 12, 14, 15.2, NURSE, z=3)     # nurse station in SW corner
    ax.add_patch(mpatches.Rectangle((11.3, 14.1), 2.2, 0.6, facecolor="#dfeceb", edgecolor=WALL, lw=0.5, zorder=5))
    ax.text(12.5, 13.1, "NURSE", ha="center", va="center", fontsize=6.4, fontweight="bold", zorder=9)
    # perimeter beds (heads to wall), clear of the 4 doors + nurse corner
    bed(ax, 11.3, 24.0, 2.1, 1.6, GENBED, "top"); bed(ax, 16.6, 24.0, 2.1, 1.6, GENBED, "top")   # north
    bed(ax, 16.6, 12.2, 2.1, 1.6, GENBED, "bottom")                                              # south (east of door)
    bed(ax, 16.9, 15.6, 2.0, 1.7, GENBED, "right"); bed(ax, 16.9, 21.2, 2.0, 1.7, GENBED, "right")  # east
    bed(ax, 11.1, 15.6, 2.0, 1.7, GENBED, "left"); bed(ax, 11.1, 21.2, 2.0, 1.7, GENBED, "left")    # west
    mejhe(ax, 14.4, 18.2, 1.2, 0.55); cluster(ax, 15.0, 19.6, 3, s=0.7)
    walls(ax, gx0, gy0, gx1, gy1, gates=[("S", 14.0, 16.0), ("N", 14.0, 16.0), ("W", 18.0, 20.0), ("E", 18.0, 20.0)])
    ax.text(15, 24.9, "GENERAL WARD", ha="center", fontsize=7.6, fontweight="bold", zorder=9)
    ax.text(15, 16.3, "(4 entrances)", ha="center", fontsize=6.4, style="italic", zorder=9)

    # attendants camped in the surrounding circulation (density)
    for (cx, cy) in [(9.6, 8.5), (20.4, 8.5), (9.6, 15), (20.4, 15), (9.6, 22), (20.4, 22),
                     (9.6, 28.5), (20.4, 28.5), (15, 9), (15, 28.5)]:
        cluster(ax, cx, cy, 3)

    # ================= TOP centre: mosque / exit / ICU =======================
    fill(ax, 8, 31, 13, 40, MOSQUE, z=2); walls(ax, 8, 31, 13, 40, gates=[("S", 10, 11.5)])
    ax.text(10.5, 37, "MOSQUE", ha="center", fontsize=8, fontweight="bold", color="#14532d", zorder=9)
    ax.text(10.5, 36, "Namaz Ghor\n(male-only)", ha="center", fontsize=5.8, style="italic", color="#14532d", zorder=9)
    fill(ax, 8, 31, 10, 32.4, WUDU, z=3); walls(ax, 8, 31, 10, 32.4, gates=[("N", 8.5, 9.7)])
    ax.text(9, 31.7, "Wudu", ha="center", va="center", fontsize=5.6, zorder=9)
    fill(ax, 17, 31, 22, 40, ICUC, z=2)
    for k in range(3):
        bed(ax, 17.3, 31.4 + k * 2.7, 1.7, 2.1, ICUBED, "left")
    walls(ax, 17, 31, 22, 40, gates=[("S", 19, 20.5)])
    ax.text(20.5, 39.2, "ICU", ha="center", fontsize=7.6, fontweight="bold", zorder=9)

    # ================= BOTTOM: emergency / pharmacy / store ==================
    fill(ax, 0, 0, 8, 5, EMERG, z=2)
    for k in range(2):
        bed(ax, 0.7 + k * 3.4, 0.7, 2.9, 1.8, np.array([0.84, 0.33, 0.30]), "bottom")
    walls(ax, 0, 0, 8, 5, gates=[("N", 3.0, 5.0)])
    ax.text(4, 4.4, "EMERGENCY (Joruri)", ha="center", fontsize=6.8, fontweight="bold", zorder=9)
    ax.add_patch(mpatches.Rectangle((8.15, 0.6), 1.5, 1.1, facecolor="#8a8f98", edgecolor=WALL, lw=1.0, zorder=6))
    ax.text(8.9, 1.1, "Ansar", ha="center", va="center", fontsize=5.4, color="white", fontweight="bold", zorder=9)
    for ty in (2.4, 3.6):
        ax.add_patch(mpatches.Rectangle((8.2, ty), 1.4, 0.55, facecolor="#c9cdd6", edgecolor=WALL, lw=0.5, zorder=6))
    ax.text(8.9, 4.3, "trolley", fontsize=5.2, style="italic", color="#444", zorder=9)
    fill(ax, 19, 0, 25, 5, PHARM, z=2); walls(ax, 19, 0, 25, 5, gates=[("N", 21, 23)])
    ax.text(22, 2.3, "PHARMACY\n(Oushud)", ha="center", va="center", fontsize=6.6, fontweight="bold", zorder=9)
    fill(ax, 25, 0, 30, 5, SERV, z=2); walls(ax, 25, 0, 30, 5, gates=[("N", 26.5, 28.5)])
    ax.text(27.5, 2.3, "STORE\nROOM", ha="center", va="center", fontsize=6.6, fontweight="bold", zorder=9)

    # ================= outer wall + entrances/exit ===========================
    walls(ax, 0, 0, 30, 40, gates=[("S", 3.0, 5.0), ("S", 12.5, 16.5), ("N", 13.5, 16.5)], lw=4)
    ax.annotate("TO RADIOLOGY,\nWARDS & OT", xy=(15, 40), xytext=(15, 41.4), ha="center", fontsize=7.4,
                fontweight="bold", color="#333", arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5), zorder=9)
    ax.annotate("AMBULANCE\nENTRANCE", xy=(4, 0), xytext=(4, -1.7), ha="center", fontsize=7,
                fontweight="bold", color="#333", arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4), zorder=9)
    ax.annotate("MAIN ENTRANCE", xy=(14.5, 0), xytext=(14.5, -1.6), ha="center", fontsize=7.2,
                fontweight="bold", color="#333", arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4), zorder=9)

    handles = [
        mpatches.Patch(facecolor=MALEBED, label="Male-ward bed"),
        mpatches.Patch(facecolor=FEMBED, label="Female-ward bed"),
        mpatches.Patch(facecolor=GENBED, label="General-ward bed (perimeter)"),
        mpatches.Patch(facecolor=OVER, label="Vertical overflow bed"),
        mpatches.Patch(facecolor=ICUBED, label="ICU bed"),
        mpatches.Patch(facecolor=MEJHE, label="Mejhe (floor mattress)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ATTEND, markersize=8, label="Family attendant"),
        mpatches.Patch(facecolor=MOSQUE, label="Mosque + Wudu"),
        mpatches.Patch(facecolor=NURSE, label="Nurse (ward corner)"),
        mpatches.Patch(facecolor=INTERN, label="Intern doctor room"),
        mpatches.Patch(facecolor=STAFFR, label="Staff room"),
        mpatches.Patch(facecolor=EMERG, label="Emergency"),
        mpatches.Patch(facecolor=PHARM, label="Pharmacy"),
        mpatches.Patch(facecolor=SERV, label="Store room"),
        Line2D([0], [0], color=WALL, lw=3, label="Wall (gaps = gates)"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.2, frameon=True, title="Legend")

    ax.set_xlim(-2.5, 30.5); ax.set_ylim(-2.5, 42)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("LMIC-HospitalNav-v1 — DMCH redesign v4  (30 x 40 portrait, CONCEPT DRAFT)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = ROOT / "experiments" / "prayer_experiment" / f"redesign_bd4.{ext}"
        fig.savefig(out, dpi=165, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
