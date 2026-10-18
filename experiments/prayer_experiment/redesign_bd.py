"""BANGLADESH-context DMCH ward-floor redesign — CONCEPT schematic (DRAFT).

Extends the fused layout (central nurse hub, male-only mosque side room, dense
gender wards, corridor overflow + Mejhe, service rooms, entrances) with the
Bangladeshi realities that were still missing:

  * FAMILY ATTENDANTS (Ayah / relatives) camped at the bedside AND on the
    corridor floors — the dominant pedestrian crowd and the paper's
    attendant-proxemics cultural constraint.
  * WUDU (ablution / Ozu) area + SHOE RACK at the mosque.
  * TICKET / registration QUEUE snaking from the ambulatory entrance.
  * ANSAR (security guard) post at the ambulance entrance.
  * TROLLEY / stretcher + wheelchair bay.
  * Romanized Bangla sublabels (Purush/Mohila Ward, Namaz Ghor, Joruri Bibhag).

Pure matplotlib (no env). Writes prayer_experiment/redesign_bd.png/.pdf.
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

WALL   = np.array([0.25, 0.26, 0.30])
CORR   = np.array([0.90, 0.91, 0.93])
MALEBG = np.array([0.83, 0.91, 0.98]); MALEBED = np.array([0.30, 0.52, 0.74])
FEMBG  = np.array([0.99, 0.90, 0.90]); FEMBED  = np.array([0.83, 0.47, 0.55])
MOSQUE = np.array([0.56, 0.83, 0.50]); WUDU = np.array([0.66, 0.86, 0.88])
OVER   = np.array([0.96, 0.69, 0.26])
CRIT   = np.array([0.84, 0.33, 0.30]); CRITBG = np.array([0.99, 0.87, 0.85])
MEJHE  = np.array([0.93, 0.90, 0.48])
SERV   = np.array([0.80, 0.80, 0.84])
NURSE  = np.array([0.40, 0.76, 0.73])
TRIAGE = np.array([0.62, 0.68, 0.87])
EMERG  = np.array([0.95, 0.61, 0.57])
PHARM  = np.array([0.60, 0.67, 0.86])
ATTEND = "#6a51a3"     # family attendants (Ayah / relatives)


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
                 boxstyle="round,pad=0,rounding_size=0.16", facecolor=c, edgecolor=WALL, lw=0.5, zorder=4))
    pw = 0.36
    if head in ("left", "right"):
        px = x + 0.12 if head == "left" else x + w - pw - 0.12
        ax.add_patch(mpatches.Rectangle((px, y + 0.2), pw, h - 0.4, facecolor="white", edgecolor=WALL, lw=0.4, zorder=5))
    else:
        py = y + 0.12 if head == "bottom" else y + h - pw - 0.12
        ax.add_patch(mpatches.Rectangle((x + 0.2, py), w - 0.4, pw, facecolor="white", edgecolor=WALL, lw=0.4, zorder=5))


def bed_col(ax, x, y0, n, dy, w, h, c, head):
    for i in range(n):
        bed(ax, x, y0 + i * dy, w, h, c, head)


def mejhe(ax, x, y0, n, dy):
    for i in range(n):
        ax.add_patch(mpatches.Rectangle((x, y0 + i * dy), 1.35, 0.62, facecolor=MEJHE, edgecolor=WALL, lw=0.3, zorder=4))


def person(ax, x, y, c=ATTEND, s=1.0, z=8, a=0.9):
    ax.add_patch(mpatches.Circle((x, y + 0.26 * s), 0.15 * s, facecolor=c, edgecolor="none", zorder=z, alpha=a))
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.15 * s, y - 0.16 * s), 0.30 * s, 0.40 * s,
                 boxstyle="round,pad=0,rounding_size=0.09", facecolor=c, edgecolor="none", zorder=z, alpha=a))


_OFF = [(-0.35, 0.0), (0.35, 0.05), (0.0, -0.3), (-0.3, -0.35), (0.3, -0.4), (-0.05, 0.35)]


def cluster(ax, cx, cy, n, s=1.0, c=ATTEND):
    for i in range(min(n, len(_OFF))):
        dx, dy = _OFF[i]
        person(ax, cx + dx, cy + dy, c=c, s=s)


def main():
    fig, ax = plt.subplots(figsize=(15, 11.4))
    fill(ax, 0, 0, 40, 30, CORR, z=0)

    # ---- MALE WARD (Purush) : dense beds + bedside attendants ----------------
    fill(ax, 0, 4, 9, 25.5, MALEBG)
    bed_col(ax, 0.4, 4.4, 8, 2.55, 2.2, 1.7, MALEBED, "left")
    bed_col(ax, 6.4, 4.4, 8, 2.55, 2.2, 1.7, MALEBED, "right")
    mejhe(ax, 3.7, 5.0, 8, 2.55)
    for i in range(8):
        cluster(ax, 4.5, 5.35 + i * 2.55, 2, s=0.85)     # attendants camped in the aisle
    walls(ax, 0, 4, 9, 25.5, gates=[("E", 5.5, 7.5), ("E", 20.5, 22.5), ("E", 14.0, 15.5)])
    ax.text(4.5, 24.9, "MALE WARD", ha="center", fontsize=9, fontweight="bold", zorder=9)
    ax.text(4.5, 24.0, "Purush Ward (gender-segregated)", ha="center", fontsize=7, style="italic", zorder=9)

    # ---- FEMALE WARD (Mohila) : mirror --------------------------------------
    fill(ax, 31, 4, 40, 25.5, FEMBG)
    bed_col(ax, 31.4, 4.4, 8, 2.55, 2.2, 1.7, FEMBED, "left")
    bed_col(ax, 37.4, 4.4, 8, 2.55, 2.2, 1.7, FEMBED, "right")
    mejhe(ax, 35.05, 5.0, 8, 2.55)
    for i in range(8):
        cluster(ax, 35.6, 5.35 + i * 2.55, 2, s=0.85)
    walls(ax, 31, 4, 40, 25.5, gates=[("W", 5.5, 7.5), ("W", 20.5, 22.5)])
    ax.text(35.5, 24.9, "FEMALE WARD", ha="center", fontsize=9, fontweight="bold", zorder=9)
    ax.text(35.5, 24.0, "Mohila Ward (gender-segregated)", ha="center", fontsize=7, style="italic", zorder=9)

    # ---- MOSQUE (Namaz Ghor, male-only) + WUDU + shoe rack ------------------
    fill(ax, 9, 11.8, 14, 17.8, MOSQUE, z=2)
    walls(ax, 9, 11.8, 14, 17.8, gates=[("E", 13.5, 15.5), ("W", 13.0, 14.3)])
    ax.text(11.5, 16.1, "MOSQUE", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#14532d", zorder=9)
    ax.text(11.5, 15.0, "Namaz Ghor\n(male-only)", ha="center", va="center", fontsize=6.8, style="italic", color="#14532d", zorder=9)
    # wudu (ablution) alcove attached below the mosque
    fill(ax, 9, 10.0, 12.5, 11.8, WUDU, z=2); walls(ax, 9, 10.0, 12.5, 11.8, gates=[("N", 10.2, 11.5)])
    ax.text(10.75, 10.85, "Wudu\n(Ozu)", ha="center", va="center", fontsize=6.6, zorder=9)
    # shoe rack at the mosque door
    ax.add_patch(mpatches.Rectangle((14.05, 13.4), 0.5, 2.2, facecolor="#b58a5a", edgecolor=WALL, lw=0.5, hatch="---", zorder=6))
    ax.text(15.3, 16.2, "shoe rack", fontsize=6, style="italic", color="#5a4a2a", zorder=9)
    cluster(ax, 14.9, 14.6, 3, s=0.8)   # men entering to pray

    # ---- NURSE STATION (central hub / spawn) --------------------------------
    fill(ax, 16, 12, 23, 17.8, NURSE, z=2)
    ax.add_patch(mpatches.Rectangle((16.6, 12.6), 5.8, 4.6, facecolor="none", edgecolor=WALL, lw=1.4, zorder=6))
    for dx in (17.2, 20.2):
        ax.add_patch(mpatches.Rectangle((dx, 16.4), 2.2, 0.7, facecolor="#dfeceb", edgecolor=WALL, lw=0.5, zorder=7))
    ax.text(19.5, 14.5, "NURSE\nSTATION\n(central hub)", ha="center", va="center", fontsize=8.4, fontweight="bold", zorder=9)

    # ---- corridor OVERFLOW beds + attendants camping the circulation --------
    bed_col(ax, 14.4, 4.6, 3, 2.4, 1.9, 1.5, OVER, "left")
    bed_col(ax, 23.1, 4.6, 3, 2.4, 1.9, 1.5, OVER, "right")
    bed_col(ax, 24.6, 12.2, 2, 2.6, 1.9, 1.5, OVER, "right")
    mejhe(ax, 17.0, 5.1, 3, 2.4); mejhe(ax, 21.2, 5.1, 3, 2.4)
    for (cx, cy, n) in [(16.6, 5.6, 3), (21.9, 6.0, 2), (16.4, 10.2, 3), (23.0, 10.0, 3),
                        (26.5, 14.5, 2), (17.5, 19.0, 3), (24.5, 19.5, 3), (13.0, 20.0, 2),
                        (26.5, 8.0, 2), (20.0, 9.5, 2)]:
        cluster(ax, cx, cy, n, s=0.85)

    # ---- CRITICAL CARE / ICU -------------------------------------------------
    fill(ax, 15, 19.5, 27, 24.6, CRITBG, z=2)
    for k in range(4):
        bed(ax, 15.6 + k * 2.9, 20.0, 2.4, 2.0, CRIT, "top")
        person(ax, 16.8 + k * 2.9, 22.5, s=0.8)   # 1 attendant per bed (restricted)
    walls(ax, 15, 19.5, 27, 24.6, gates=[("S", 19.5, 22.5)])
    ax.text(21, 23.9, "CRITICAL CARE / ICU", ha="center", fontsize=8.3, fontweight="bold", zorder=9)

    # ---- BOTTOM: emergency / ticket-queue / triage / pharmacy + fixtures ----
    fill(ax, 0, 0, 9, 3.8, EMERG, z=2)
    for k in range(2):
        bed(ax, 1.2 + k * 3.6, 0.8, 3.0, 1.9, CRIT, "bottom")
    walls(ax, 0, 0, 9, 3.8, gates=[("N", 3.5, 5.5)])
    ax.text(4.5, 3.25, "EMERGENCY", ha="center", fontsize=7.8, fontweight="bold", zorder=9)
    ax.text(4.5, 2.75, "Joruri Bibhag", ha="center", fontsize=6, style="italic", zorder=9)
    # Ansar (security) post by the ambulance entrance
    ax.add_patch(mpatches.Rectangle((6.6, 3.9), 1.8, 1.2, facecolor="#8a8f98", edgecolor=WALL, lw=1.0, zorder=6))
    ax.text(7.5, 4.5, "Ansar", ha="center", va="center", fontsize=6.2, color="white", fontweight="bold", zorder=9)
    person(ax, 7.5, 5.4, c="#3a3f48", s=0.85)
    # trolley / wheelchair bay
    for tx in (9.7, 11.3):
        ax.add_patch(mpatches.Rectangle((tx, 4.2, ), 1.1, 0.55, facecolor="#c9cdd6", edgecolor=WALL, lw=0.5, zorder=6))
        ax.add_patch(mpatches.Circle((tx + 0.2, 4.2), 0.13, facecolor=WALL, zorder=7))
        ax.add_patch(mpatches.Circle((tx + 0.9, 4.2), 0.13, facecolor=WALL, zorder=7))
    ax.text(11.0, 5.2, "trolley / wheelchair bay", fontsize=6, style="italic", color="#444", zorder=9)

    fill(ax, 15, 0.4, 20, 3.8, TRIAGE, z=2); walls(ax, 15, 0.4, 20, 3.8, gates=[("N", 16.5, 18.5)])
    ax.text(17.5, 2.1, "TICKET /\nREGIST.", ha="center", va="center", fontsize=6.6, zorder=9)
    fill(ax, 20, 0.4, 26, 3.8, TRIAGE, z=2); walls(ax, 20, 0.4, 26, 3.8, gates=[("N", 22, 24)])
    ax.text(23, 2.1, "TRIAGE", ha="center", fontsize=6.8, zorder=9)
    fill(ax, 31, 0, 40, 3.8, PHARM, z=2); walls(ax, 31, 0, 40, 3.8, gates=[("N", 34, 36)])
    ax.text(35.5, 2.0, "PHARMACY", ha="center", fontsize=7.8, fontweight="bold", zorder=9)
    ax.text(35.5, 1.4, "Oushud", ha="center", fontsize=6, style="italic", zorder=9)
    # ticket queue snaking from the ambulatory entrance to the counter
    for qx in np.arange(17.5, 28.5, 0.9):
        person(ax, qx, 4.5 if qx < 22 else 4.5, s=0.7, c="#7a6aa8")
    ax.text(24.5, 5.3, "ticket queue", fontsize=6.2, style="italic", color="#4a3f6a", zorder=9)

    # ---- TOP: service rooms / supply / exit / lab ---------------------------
    fill(ax, 0, 25.5, 9, 30, SERV, z=2)
    walls(ax, 0, 25.5, 4.5, 30, gates=[("S", 1.5, 3)]); walls(ax, 4.5, 25.5, 9, 30, gates=[("S", 6, 7.5)])
    ax.text(2.25, 28, "STORAGE", ha="center", fontsize=7, zorder=9)
    ax.text(6.75, 28, "SOILED\nUTIL.", ha="center", fontsize=7, zorder=9)
    fill(ax, 9, 25.5, 15, 30, SERV, z=2); walls(ax, 9, 25.5, 15, 30, gates=[("S", 11, 13)])
    ax.text(12, 28, "SUPPLY", ha="center", fontsize=7.6, zorder=9)
    fill(ax, 24, 25.5, 31, 30, SERV, z=2); walls(ax, 24, 25.5, 31, 30, gates=[("S", 26.5, 28.5)])
    ax.text(27.5, 28, "STAFF\nLOUNGE", ha="center", fontsize=7, zorder=9)
    fill(ax, 31, 25.5, 35.5, 30, PHARM, z=2); walls(ax, 31, 25.5, 35.5, 30, gates=[("S", 32.5, 34)])
    ax.text(33.25, 28, "LAB", ha="center", fontsize=7.2, zorder=9)
    fill(ax, 35.5, 25.5, 40, 30, SERV, z=2); walls(ax, 35.5, 25.5, 40, 30, gates=[("S", 37, 38.5)])
    ax.text(37.75, 28, "MED\nSTORAGE", ha="center", fontsize=7, zorder=9)

    # ---- outer wall + entrances/exit ----------------------------------------
    walls(ax, 0, 0, 40, 30, gates=[("S", 3.5, 6.0), ("S", 26.5, 29.5), ("N", 17.5, 22.5)], lw=4)
    ax.annotate("TO LAB, RADIOLOGY,\nINPATIENT WARDS & OT", xy=(20, 30), xytext=(20, 31.4),
                ha="center", fontsize=8, fontweight="bold", color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5), zorder=9)
    ax.annotate("AMBULANCE\nENTRANCE", xy=(4.7, 0), xytext=(4.7, -2.2), ha="center", fontsize=7.6,
                fontweight="bold", color="#333", arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5), zorder=9)
    ax.annotate("MAIN (AMBULATORY)\nENTRANCE", xy=(28, 0), xytext=(28, -2.2), ha="center", fontsize=7.6,
                fontweight="bold", color="#333", arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5), zorder=9)

    # ---- annotations ---------------------------------------------------------
    ax.annotate("FAMILY ATTENDANTS (Ayah / relatives) camp at every bedside and on the\n"
                "corridor floors — the dominant crowd in a Bangladeshi ward, and the\n"
                "paper's attendant-proxemics constraint the robot must respect.",
                xy=(16.6, 5.9), xytext=(2.0, -2.3), ha="left", fontsize=8, color="#4a2f6a",
                arrowprops=dict(arrowstyle="-|>", color="#6a51a3", lw=1.4),
                bbox=dict(boxstyle="round,pad=0.32", fc="#f1ecf7", ec="#6a51a3", lw=1.1), zorder=12)
    ax.annotate("Male-only MOSQUE (Namaz Ghor) with Wudu + shoe rack —\n"
                "a realistic side room off the corridor by the nurse hub.",
                xy=(11.5, 11.8), xytext=(0.5, 7.6), ha="left", fontsize=7.6, color="#14532d",
                arrowprops=dict(arrowstyle="-|>", color="#14532d", lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", fc="#eaf6ea", ec="#1b7837", lw=1.0), zorder=12)

    handles = [
        mpatches.Patch(facecolor=MALEBED, label="Male-ward bed"),
        mpatches.Patch(facecolor=FEMBED, label="Female-ward bed"),
        mpatches.Patch(facecolor=OVER, label="Corridor overflow bed"),
        mpatches.Patch(facecolor=CRIT, label="Critical / emergency bed"),
        mpatches.Patch(facecolor=MEJHE, label="Mejhe (floor mattress)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ATTEND, markersize=9, label="Family attendant (Ayah)"),
        mpatches.Patch(facecolor=MOSQUE, label="Mosque (Namaz Ghor)"),
        mpatches.Patch(facecolor=WUDU, label="Wudu (ablution)"),
        mpatches.Patch(facecolor=NURSE, label="Nurse station (hub)"),
        mpatches.Patch(facecolor=EMERG, label="Emergency"),
        mpatches.Patch(facecolor=PHARM, label="Pharmacy / Lab"),
        mpatches.Patch(facecolor=TRIAGE, label="Ticket / Triage"),
        mpatches.Patch(facecolor=SERV, label="Service / Supply"),
        mpatches.Patch(facecolor=CORR, label="Circulation"),
        Line2D([0], [0], color=WALL, lw=3, label="Wall (gaps = gated doors)"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8.2,
              frameon=True, title="Legend")

    ax.set_xlim(-3.2, 40.5); ax.set_ylim(-3, 32)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("LMIC-HospitalNav-v1 — DMCH redesign with Bangladeshi context  (CONCEPT DRAFT)\n"
                 "central nurse hub · male-only mosque + wudu · dense wards · corridor overflow · "
                 "family attendants · ticket queue · Ansar",
                 fontsize=11.8, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = ROOT / "experiments" / "prayer_experiment" / f"redesign_bd.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
