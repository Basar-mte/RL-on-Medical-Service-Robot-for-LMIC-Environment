"""DMCH redesign v3 — realism pass. CONCEPT schematic (DRAFT).

Changes vs v2:
  * Radiology room REMOVED (had no valid entrance).
  * Storage + Soiled Utility (by male ward) REMOVED; intern/staff rooms grow to
    fill the top corners.
  * NO bed across any ward entrance — every gate kept clear.
  * Ticket Registry REMOVED -> open circulation from the main entrance.
  * NURSE STATION moved into a corner of the central GENERAL WARD; beds lined
    AROUND the general-ward perimeter.
  * Mosque (former supply) + wudu, ICU (former staff lounge), ward intern/staff
    rooms, vertical corridor-overflow beds, emergency/Ansar/trolley, store room,
    pharmacy, family attendants — retained.

Pure matplotlib. Writes prayer_experiment/redesign_bd3.png/.pdf.
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
    ax.add_patch(mpatches.FancyBboxPatch((x + 0.06, y + 0.06), w - 0.12, h - 0.12,
                 boxstyle="round,pad=0,rounding_size=0.15", facecolor=c, edgecolor=WALL, lw=0.5, zorder=4))
    pw = 0.34
    if head in ("left", "right"):
        px = x + 0.12 if head == "left" else x + w - pw - 0.12
        ax.add_patch(mpatches.Rectangle((px, y + 0.18), pw, h - 0.36, facecolor="white", edgecolor=WALL, lw=0.4, zorder=5))
    else:
        py = y + 0.12 if head == "bottom" else y + h - pw - 0.12
        ax.add_patch(mpatches.Rectangle((x + 0.18, py), w - 0.36, pw, facecolor="white", edgecolor=WALL, lw=0.4, zorder=5))


def bed_col_skip(ax, x, y0, y1, pitch, w, h, c, head, skips=()):
    """Stack beds up a column, skipping any that would overlap a gate range."""
    y = y0
    while y + h <= y1 + 1e-6:
        if all(y + h <= s0 or y >= s1 for (s0, s1) in skips):
            bed(ax, x, y, w, h, c, head)
        y += pitch


def mejhe(ax, x, y0, n, dy):
    for i in range(n):
        ax.add_patch(mpatches.Rectangle((x, y0 + i * dy), 1.25, 0.58, facecolor=MEJHE, edgecolor=WALL, lw=0.3, zorder=4))


def person(ax, x, y, c=ATTEND, s=0.8, z=8, a=0.9):
    ax.add_patch(mpatches.Circle((x, y + 0.24 * s), 0.14 * s, facecolor=c, edgecolor="none", zorder=z, alpha=a))
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.14 * s, y - 0.15 * s), 0.28 * s, 0.37 * s,
                 boxstyle="round,pad=0,rounding_size=0.08", facecolor=c, edgecolor="none", zorder=z, alpha=a))


_OFF = [(-0.3, 0.0), (0.3, 0.05), (0.0, -0.28), (-0.26, -0.3), (0.28, -0.34)]


def cluster(ax, cx, cy, n, s=0.78, c=ATTEND):
    for i in range(min(n, len(_OFF))):
        dx, dy = _OFF[i]
        person(ax, cx + dx, cy + dy, c=c, s=s)


def main():
    fig, ax = plt.subplots(figsize=(15, 11.4))
    fill(ax, 0, 0, 40, 30, CORR, z=0)
    MG = [(7.0, 9.0), (15.0, 17.0)]   # ward east-wall gate ranges (kept clear)

    # ================= MALE side: ward + intern/staff (top corners) ==========
    fill(ax, 0, 4, 9, 20.3, MALEBG)
    bed_col_skip(ax, 0.4, 4.3, 20.0, 2.6, 2.1, 1.7, MALEBED, "left")
    bed_col_skip(ax, 6.5, 4.3, 20.0, 2.6, 2.1, 1.7, MALEBED, "right", skips=MG)
    for i in range(6):
        yc = 5.1 + i * 2.6
        if all(yc <= s0 or yc >= s1 for s0, s1 in MG):
            mejhe(ax, 3.75, yc, 1, 1); cluster(ax, 4.6, yc, 2)
    walls(ax, 0, 4, 9, 20.3, gates=[("E", 7.0, 9.0), ("E", 15.0, 17.0)])
    ax.text(4.5, 19.5, "MALE WARD (Purush)", ha="center", fontsize=8.4, fontweight="bold", zorder=9)
    bed_col_skip(ax, 9.15, 4.5, 19.8, 2.55, 1.5, 2.15, OVER, "top", skips=MG)   # vertical overflow, gates clear
    fill(ax, 0, 20.3, 4.5, 30, INTERN, z=2); walls(ax, 0, 20.3, 4.5, 30, gates=[("S", 1.5, 3)])
    ax.text(2.25, 25, "INTERN\nDOCTOR\nROOM", ha="center", va="center", fontsize=7, fontweight="bold", zorder=9)
    fill(ax, 4.5, 20.3, 9, 30, STAFFR, z=2); walls(ax, 4.5, 20.3, 9, 30, gates=[("S", 6, 7.5)])
    ax.text(6.75, 25, "STAFF\nROOM", ha="center", va="center", fontsize=7, fontweight="bold", zorder=9)

    # ================= FEMALE side (mirror) ==================================
    fill(ax, 31, 4, 40, 20.3, FEMBG)
    bed_col_skip(ax, 31.4, 4.3, 20.0, 2.6, 2.1, 1.7, FEMBED, "left", skips=MG)
    bed_col_skip(ax, 37.5, 4.3, 20.0, 2.6, 2.1, 1.7, FEMBED, "right")
    for i in range(6):
        yc = 5.1 + i * 2.6
        if all(yc <= s0 or yc >= s1 for s0, s1 in MG):
            mejhe(ax, 35.05, yc, 1, 1); cluster(ax, 35.6, yc, 2)
    walls(ax, 31, 4, 40, 20.3, gates=[("W", 7.0, 9.0), ("W", 15.0, 17.0)])
    ax.text(35.5, 19.5, "FEMALE WARD (Mohila)", ha="center", fontsize=8.4, fontweight="bold", zorder=9)
    bed_col_skip(ax, 29.35, 4.5, 19.8, 2.55, 1.5, 2.15, OVER, "top", skips=MG)
    fill(ax, 31, 20.3, 35.5, 30, INTERN, z=2); walls(ax, 31, 20.3, 35.5, 30, gates=[("S", 32.5, 34)])
    ax.text(33.25, 25, "INTERN\nDOCTOR\nROOM", ha="center", va="center", fontsize=7, fontweight="bold", zorder=9)
    fill(ax, 35.5, 20.3, 40, 30, STAFFR, z=2); walls(ax, 35.5, 20.3, 40, 30, gates=[("S", 37, 38.5)])
    ax.text(37.75, 25, "STAFF\nROOM", ha="center", va="center", fontsize=7, fontweight="bold", zorder=9)

    # ================= CENTER: GENERAL WARD (4 doors) + nurse corner ==========
    gx0, gy0, gx1, gy1 = 15, 9.3, 25, 19.8
    fill(ax, gx0, gy0, gx1, gy1, GENBG, z=2)
    # nurse station in the SW corner
    fill(ax, 15, 9.3, 18, 12.4, NURSE, z=3)
    ax.add_patch(mpatches.Rectangle((15.4, 11.3), 2.2, 0.7, facecolor="#dfeceb", edgecolor=WALL, lw=0.5, zorder=5))
    ax.text(16.5, 10.4, "NURSE", ha="center", va="center", fontsize=6.6, fontweight="bold", zorder=9)
    # perimeter beds (heads to the wall), avoiding the 4 doors + nurse corner
    bed(ax, 15.4, 17.9, 2.3, 1.6, GENBED, "top"); bed(ax, 21.8, 17.9, 2.3, 1.6, GENBED, "top")   # north
    bed(ax, 21.8, 9.6, 2.3, 1.6, GENBED, "bottom")                                               # south (east of door)
    bed(ax, 22.6, 10.4, 2.2, 1.7, GENBED, "right"); bed(ax, 22.6, 16.4, 2.2, 1.7, GENBED, "right")  # east
    bed(ax, 15.2, 16.4, 2.2, 1.7, GENBED, "left")                                                # west
    mejhe(ax, 19.1, 13.4, 1, 1); mejhe(ax, 20.5, 13.4, 1, 1); cluster(ax, 20.0, 15.2, 3, s=0.72)
    walls(ax, gx0, gy0, gx1, gy1, gates=[("S", 18.5, 21.5), ("N", 18.5, 21.5), ("W", 13.5, 15.5), ("E", 13.5, 15.5)])
    ax.text(20, 18.9, "GENERAL WARD", ha="center", fontsize=8.6, fontweight="bold", zorder=9)
    ax.text(20, 12.5, "(4 entrances)", ha="center", fontsize=6.8, style="italic", zorder=9)

    # ================= TOP centre: mosque / exit / ICU =======================
    fill(ax, 9, 24, 15, 30, MOSQUE, z=2); walls(ax, 9, 24, 15, 30, gates=[("S", 11, 13)])
    ax.text(12, 28, "MOSQUE", ha="center", fontsize=8.3, fontweight="bold", color="#14532d", zorder=9)
    ax.text(12, 27.1, "Namaz Ghor (male-only)", ha="center", fontsize=5.8, style="italic", color="#14532d", zorder=9)
    fill(ax, 9, 22.6, 12, 24, WUDU, z=2); walls(ax, 9, 22.6, 12, 24, gates=[("N", 10, 11.5)])
    ax.text(10.5, 23.3, "Wudu", ha="center", va="center", fontsize=6, zorder=9)
    fill(ax, 25, 25, 31, 30, ICUC, z=2)
    for k in range(3):
        bed(ax, 25.3 + k * 1.9, 25.3, 1.6, 1.9, ICUBED, "top")
    walls(ax, 25, 25, 31, 30, gates=[("S", 27, 29)])
    ax.text(28, 29.4, "ICU", ha="center", fontsize=8, fontweight="bold", zorder=9)

    # ================= BOTTOM: emergency / store / pharmacy ==================
    fill(ax, 0, 0, 9, 3.8, EMERG, z=2)
    for k in range(2):
        bed(ax, 1.2 + k * 3.6, 0.8, 3.0, 1.9, np.array([0.84, 0.33, 0.30]), "bottom")
    walls(ax, 0, 0, 9, 3.8, gates=[("N", 3.5, 5.5)])
    ax.text(4.5, 3.25, "EMERGENCY (Joruri)", ha="center", fontsize=7.2, fontweight="bold", zorder=9)
    ax.add_patch(mpatches.Rectangle((6.6, 3.9), 1.7, 1.1, facecolor="#8a8f98", edgecolor=WALL, lw=1.0, zorder=6))
    ax.text(7.45, 4.45, "Ansar", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold", zorder=9)
    for tx in (9.7, 11.2):
        ax.add_patch(mpatches.Rectangle((tx, 4.2), 1.0, 0.5, facecolor="#c9cdd6", edgecolor=WALL, lw=0.5, zorder=6))
    ax.text(11.2, 5.1, "trolley bay", fontsize=5.8, style="italic", color="#444", zorder=9)
    fill(ax, 26, 0.4, 31, 3.8, SERV, z=2); walls(ax, 26, 0.4, 31, 3.8, gates=[("N", 27.5, 29.5)])
    ax.text(28.5, 2.1, "STORE\nROOM", ha="center", va="center", fontsize=6.8, fontweight="bold", zorder=9)
    fill(ax, 31, 0, 40, 3.8, PHARM, z=2); walls(ax, 31, 0, 40, 3.8, gates=[("N", 34, 36)])
    ax.text(35.5, 1.9, "PHARMACY (Oushud)", ha="center", fontsize=7.2, fontweight="bold", zorder=9)
    # attendants camped in the open circulation (density)
    for (cx, cy) in [(12.5, 6.5), (27.5, 6.5), (12.5, 12.0), (27.5, 12.0), (20, 6.0), (12.5, 18.0), (27.5, 18.0)]:
        cluster(ax, cx, cy, 3)

    # ================= outer wall + entrances/exit ===========================
    walls(ax, 0, 0, 40, 30, gates=[("S", 3.5, 6.0), ("S", 17.5, 22.5), ("N", 17.5, 22.5)], lw=4)
    ax.annotate("TO RADIOLOGY, INPATIENT\nWARDS & OT", xy=(20, 30), xytext=(20, 31.4),
                ha="center", fontsize=8, fontweight="bold", color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5), zorder=9)
    ax.annotate("AMBULANCE\nENTRANCE", xy=(4.7, 0), xytext=(4.7, -2.1), ha="center", fontsize=7.4,
                fontweight="bold", color="#333", arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5), zorder=9)
    ax.annotate("MAIN ENTRANCE", xy=(20, 0), xytext=(20, -1.9), ha="center", fontsize=7.6,
                fontweight="bold", color="#333", arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5), zorder=9)

    handles = [
        mpatches.Patch(facecolor=MALEBED, label="Male-ward bed"),
        mpatches.Patch(facecolor=FEMBED, label="Female-ward bed"),
        mpatches.Patch(facecolor=GENBED, label="General-ward bed (perimeter)"),
        mpatches.Patch(facecolor=OVER, label="Vertical overflow bed"),
        mpatches.Patch(facecolor=ICUBED, label="ICU bed"),
        mpatches.Patch(facecolor=MEJHE, label="Mejhe (floor mattress)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ATTEND, markersize=9, label="Family attendant"),
        mpatches.Patch(facecolor=MOSQUE, label="Mosque + Wudu"),
        mpatches.Patch(facecolor=NURSE, label="Nurse station (ward corner)"),
        mpatches.Patch(facecolor=INTERN, label="Intern doctor room"),
        mpatches.Patch(facecolor=STAFFR, label="Staff room"),
        mpatches.Patch(facecolor=EMERG, label="Emergency"),
        mpatches.Patch(facecolor=PHARM, label="Pharmacy"),
        mpatches.Patch(facecolor=SERV, label="Store room"),
        mpatches.Patch(facecolor=CORR, label="Circulation"),
        Line2D([0], [0], color=WALL, lw=3, label="Wall (gaps = gated doors)"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7.9, frameon=True, title="Legend")

    ax.set_xlim(-3.2, 40.5); ax.set_ylim(-3, 32)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("LMIC-HospitalNav-v1 — DMCH redesign v3 (realism pass)  CONCEPT DRAFT\n"
                 "nurse station in general-ward corner · beds around the ward · clear entrances · "
                 "radiology/storage removed · ticket registry removed",
                 fontsize=11.4, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = ROOT / "experiments" / "prayer_experiment" / f"redesign_bd3.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
