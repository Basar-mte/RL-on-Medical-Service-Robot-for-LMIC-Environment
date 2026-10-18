"""Draw the hierarchical-stack overview figure (manuscript/figures/fig_stack.pdf):
map+zones -> culturally-costed Dijkstra planner -> local driver (learned / residual /
classical) -> swept-path shield -> robot, with the CARS terms feeding both the reward
and the planner costs. Pure matplotlib, vector PDF, grayscale-safe."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10.2, 2.9))
ax.set_xlim(0, 102); ax.set_ylim(0, 29); ax.axis("off")


def box(x, y, w, h, title, lines, fc="#f2f2f2", lw=1.1, title_fs=8.6, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.9",
                                fc=fc, ec="black", lw=lw))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 4.6 - 2.55 * i, ln, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, label=None, dy=1.4, style="-|>", ls="-", fs=7.2, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 lw=lw, color="black", linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2, max(y1, y2) + dy, label, ha="center", va="bottom", fontsize=fs)


# ---- blocks ------------------------------------------------------------------
box(1, 9, 15.5, 13, "Map + zones", ["occupancy grid (SLAM", "on the robot), zone", "labels, prayer clock"])
box(22, 9, 20, 13, "Route planner (C2)",
    [r"$m{=}1{+}\kappa_{\rm c}p(x)/r_0$", r"Dijkstra $\Rightarrow J^{*},L^{*}$", r"waypoint @ $\ell{=}2.5$ m, LOS"])
box(47.5, 9, 22, 13, "Local driver (C4)",
    ["SAC / TD3 (CARS reward)", r"residual: $a_{\rm SF}+\frac{1}{2}a_{\pi}$", "or classical social force"])
box(75, 9, 17, 13, "Shield v2 (C3)",
    ["swept-path speed", r"governor $s_{\rm free}(\delta)$", "steer alternatives"])
box(96, 9.8, 5.2, 11.4, "", [], fc="white", lw=1.1)
ax.text(98.6, 15.5, "robot\n10 Hz", ha="center", va="center", fontsize=7.6)

# ---- main data path ----------------------------------------------------------
arrow(16.5, 15.5, 22, 15.5)
arrow(42, 15.5, 47.5, 15.5); ax.text(44.75, 17.2, r"waypoint, $L^{*}$", ha="center", va="bottom", fontsize=7.2)
arrow(69.5, 15.5, 75, 15.5, r"$(v,\delta)$")
arrow(92, 15.5, 96, 15.5)

# ---- sensing into the driver and the shield ----------------------------------
ax.text(58.5, 3.0, "on-board sensing: 48-ray LiDAR, 5 tracked pedestrians (+velocities), pose, context",
        ha="center", va="center", fontsize=7.4, style="italic")
arrow(55, 4.4, 57, 9, style="-|>", lw=1.0)
arrow(81, 4.4, 83, 9, style="-|>", lw=1.0)

# ---- CARS terms feed both levels (training-time reward, planning-time costs) --
box(26.5, 23.6, 34, 4.6, "", [], fc="#e8eef7", lw=1.0)
ax.text(43.5, 25.9, r"CARS cultural terms $p(x)$: prayer $\times\,\iota(t)$, gender wards, attendants",
        ha="center", va="center", fontsize=7.4)
arrow(33, 23.4, 32, 22.3, style="-|>", ls=(0, (3, 2)), lw=1.0)
arrow(54, 23.4, 56, 22.3, style="-|>", ls=(0, (3, 2)), lw=1.0)
ax.text(29.8, 23.3, "route costs", ha="right", va="center", fontsize=6.8)
ax.text(58.4, 23.3, "reward terms (training)", ha="left", va="center", fontsize=6.8)

plt.tight_layout(pad=0.25)
plt.savefig(OUT / "fig_stack.pdf")
plt.savefig(OUT / "fig_stack.png", dpi=220)
print("wrote", OUT / "fig_stack.pdf")
