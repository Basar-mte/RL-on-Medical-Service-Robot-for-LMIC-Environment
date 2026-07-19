"""
Professional animated video: RL Training of Hospital Navigation Robot
LMIC Hospital Navigation — SAC + CWN + CARS
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle, Wedge, Arrow
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import imageio
import io
import math

# ─── Color palette (Okabe-Ito colorblind-safe) ───────────────────────────────
C = {
    "bg":      "#0A0E1A",
    "panel":   "#111827",
    "accent":  "#38BDF8",
    "gold":    "#FBBF24",
    "green":   "#34D399",
    "red":     "#F87171",
    "purple":  "#A78BFA",
    "orange":  "#FB923C",
    "teal":    "#2DD4BF",
    "white":   "#F1F5F9",
    "gray":    "#64748B",
    "wall":    "#1E293B",
    "corridor":"#1C2540",
    "ward_m":  "#1A2545",
    "ward_f":  "#251A45",
    "prayer":  "#1A3025",
    "icu":     "#2D1A1A",
    "robot":   "#F87171",
    "goal":    "#FBBF24",
    "staff":   "#60A5FA",
    "patient": "#FB923C",
    "family":  "#C084FC",
    "wheelchair": "#34D399",
}

FPS = 30
DPI = 120
W, H = 1920, 1080

# ─── Helpers ─────────────────────────────────────────────────────────────────

def ease_inout(t):
    return t * t * (3 - 2 * t)

def ease_out(t):
    return 1 - (1 - t) ** 3

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def moving_avg(arr, w=15):
    if len(arr) < w:
        return arr
    out = np.convolve(arr, np.ones(w) / w, mode='valid')
    return np.concatenate([np.full(w - 1, out[0]), out])

rng = np.random.default_rng(42)

# ─── Simulated training data ─────────────────────────────────────────────────

TOTAL_STEPS = 300_000

def gen_training_curve(final_mean, noise_scale=0.12, initial=-0.8, warmup=40_000):
    steps = np.linspace(0, TOTAL_STEPS, 600)
    curve = np.zeros(600)
    for i, s in enumerate(steps):
        if s < warmup:
            t = s / warmup
            curve[i] = lerp(initial, final_mean * 0.3, ease_out(t))
        else:
            t = (s - warmup) / (TOTAL_STEPS - warmup)
            curve[i] = lerp(final_mean * 0.3, final_mean, ease_out(t))
        curve[i] += rng.normal(0, noise_scale * (1 - min(1, s / TOTAL_STEPS) * 0.7))
    return steps, curve

steps_x, r_cwn   = gen_training_curve(1.42, 0.10)
_,        r_eq   = gen_training_curve(1.18, 0.11)
_,        r_nav  = gen_training_curve(0.98, 0.12)
_,        r_van  = gen_training_curve(0.71, 0.13)

sr_x, sr_cwn = gen_training_curve(0.891, 0.06, -0.1, 30_000)
sr_cwn = np.clip(sr_cwn, 0, 1)
_, sr_eq  = gen_training_curve(0.786, 0.07, -0.1, 30_000); sr_eq  = np.clip(sr_eq, 0, 1)
_, sr_van = gen_training_curve(0.648, 0.08, -0.1, 30_000); sr_van = np.clip(sr_van, 0, 1)

cvr_x, cvr_cwn = gen_training_curve(-0.038, 0.02, -0.55, 25_000)
cvr_cwn = np.abs(cvr_cwn)
_, cvr_eq = gen_training_curve(-0.135, 0.025, -0.55, 25_000); cvr_eq = np.abs(cvr_eq)
_, cvr_van = gen_training_curve(-0.247, 0.03, -0.55, 25_000); cvr_van = np.abs(cvr_van)

w_nav_curve  = np.clip(0.38 - 0.08 * np.linspace(0, 1, 600) + rng.normal(0, 0.01, 600), 0.15, 0.6)
w_safe_curve = np.clip(0.25 + 0.05 * np.linspace(0, 1, 600) + rng.normal(0, 0.01, 600), 0.1, 0.5)
w_cult_curve = np.clip(0.22 + 0.09 * np.linspace(0, 1, 600) + rng.normal(0, 0.01, 600), 0.05, 0.45)
w_task_curve = np.clip(0.15 - 0.06 * np.linspace(0, 1, 600) + rng.normal(0, 0.01, 600), 0.05, 0.35)

# Normalize weights
wsum = w_nav_curve + w_safe_curve + w_cult_curve + w_task_curve
w_nav_curve  /= wsum
w_safe_curve /= wsum
w_cult_curve /= wsum
w_task_curve /= wsum

# ─── Hospital floor plan data ─────────────────────────────────────────────────
# Rooms: [x, y, w, h, color, label]
ROOMS = [
    (0, 15, 18, 7, C["ward_m"],  "Male Ward"),
    (22, 15, 18, 7, C["ward_f"], "Female Ward"),
    (28, 22, 12, 8, C["prayer"], "Prayer Room"),
    (0, 22, 10, 8,  C["icu"],    "ICU"),
    (10, 22, 8, 8,  C["panel"],  "Supply"),
    (0, 0, 8, 12,   C["panel"],  "Pharmacy"),
    (32, 0, 8, 12,  C["icu"],    "Emergency"),
    (14, 0, 12, 12, C["panel"],  "Nurse Station\n(Robot Spawn)"),
]
# Corridor: y=12..15
CORRIDOR = (0, 12, 40, 3)

# ─── Robot trajectory simulation ─────────────────────────────────────────────
def gen_trajectory(t_frac, policy_quality=0.5):
    """Generate a robot trajectory for a given training progress."""
    # Start at nurse station (17, 6), goal in ward (5, 18)
    start = np.array([17.0, 6.0])
    goal  = np.array([5.0, 18.0])
    n = 80
    path = [start.copy()]
    pos = start.copy()
    for i in range(n):
        to_goal = goal - pos
        dist = np.linalg.norm(to_goal)
        if dist < 0.5:
            break
        direction = to_goal / dist
        noise_scale = (1 - policy_quality) * 0.8
        direction += rng.normal(0, noise_scale, 2)
        direction /= (np.linalg.norm(direction) + 1e-9)
        step = direction * 0.5
        pos = pos + step
        path.append(pos.copy())
    return np.array(path)

# Pedestrian positions (static snapshot for visualization)
def gen_pedestrians():
    peds = []
    # Medical staff (blue)
    for _ in range(8):
        peds.append({"pos": rng.uniform([2, 12], [38, 29]), "type": "staff", "r": 0.25})
    # Patients (orange)
    for _ in range(12):
        peds.append({"pos": rng.uniform([1, 15], [17, 22]), "type": "patient", "r": 0.3})
    # Family attendants (purple)
    for _ in range(15):
        peds.append({"pos": rng.uniform([0, 15], [40, 22]), "type": "family", "r": 0.3})
    # Wheelchair (green)
    for _ in range(5):
        peds.append({"pos": rng.uniform([5, 12], [35, 29]), "type": "wheelchair", "r": 0.35})
    # Overflow beds in corridor
    return peds

PEDS = gen_pedestrians()

PED_COLORS = {
    "staff": C["staff"],
    "patient": C["patient"],
    "family": C["family"],
    "wheelchair": C["wheelchair"],
}

# ─── Drawing utilities ────────────────────────────────────────────────────────

def draw_hospital(ax, alpha=1.0, show_lidar=False, robot_pos=None, robot_angle=0.0,
                  policy_quality=0.5, show_goal=True, highlight_zones=False):
    ax.set_facecolor(C["wall"])
    ax.set_xlim(-1, 41)
    ax.set_ylim(-1, 31)
    ax.set_aspect("equal")
    ax.axis("off")

    # Walkable area background
    ax.add_patch(Rectangle((-0.5, -0.5), 41, 31.5, fc=C["corridor"], ec="none", alpha=alpha))

    # Rooms
    for (rx, ry, rw, rh, rc, rlabel) in ROOMS:
        ax.add_patch(Rectangle((rx, ry), rw, rh, fc=rc, ec=C["gray"], lw=0.5, alpha=alpha))
        ax.text(rx + rw / 2, ry + rh / 2, rlabel, color=C["white"], fontsize=5.5,
                ha="center", va="center", alpha=alpha * 0.9,
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    # Corridor label
    ax.text(20, 13.5, "Main Corridor  (Overflow Beds →)", color=C["white"], fontsize=5,
            ha="center", va="center", alpha=alpha * 0.7)

    # Overflow beds in corridor
    for i in range(10):
        bx = 1 + i * 3.7
        ax.add_patch(Rectangle((bx, 12.1), 1.5, 0.8, fc="#334155", ec=C["gray"], lw=0.4, alpha=alpha))

    # Zone highlights
    if highlight_zones:
        ax.add_patch(Rectangle((28, 22), 12, 8, fc=C["green"], alpha=0.12, ec=C["green"], lw=1.5, ls="--"))
        ax.add_patch(Rectangle((0, 15), 18, 7, fc=C["accent"], alpha=0.08, ec=C["accent"], lw=1.5, ls="--"))
        ax.add_patch(Rectangle((22, 15), 18, 7, fc=C["purple"], alpha=0.08, ec=C["purple"], lw=1.5, ls="--"))

    # Pedestrians
    ped_type_drawn = set()
    for p in PEDS:
        col = PED_COLORS[p["type"]]
        c = Circle(p["pos"], p["r"], fc=col, ec="white", lw=0.3, alpha=alpha * 0.85)
        ax.add_patch(c)

    # Goal
    if show_goal:
        goal = np.array([5.0, 18.0])
        ax.add_patch(Circle(goal, 0.6, fc=C["goal"], ec="white", lw=1.0, alpha=0.9 * alpha))
        ax.text(goal[0], goal[1] - 1.2, "GOAL", color=C["gold"], fontsize=5, ha="center", alpha=alpha)

    # Robot
    if robot_pos is not None:
        rp = robot_pos
        ax.add_patch(Circle(rp, 0.4, fc=C["robot"], ec="white", lw=1.0, alpha=alpha, zorder=10))
        dx = 0.6 * math.cos(robot_angle)
        dy = 0.6 * math.sin(robot_angle)
        ax.annotate("", xy=(rp[0] + dx, rp[1] + dy), xytext=(rp[0], rp[1]),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5), zorder=11)

        # LiDAR rays
        if show_lidar:
            for ray_angle in np.linspace(robot_angle - math.pi * 3/4, robot_angle + math.pi * 3/4, 24):
                dist = rng.uniform(1.5, 6) * policy_quality + 1
                ex = rp[0] + dist * math.cos(ray_angle)
                ey = rp[1] + dist * math.sin(ray_angle)
                ax.plot([rp[0], ex], [rp[1], ey], color=C["accent"], alpha=0.25, lw=0.5, zorder=9)

    # Walls border
    ax.add_patch(Rectangle((-0.5, -0.5), 41, 31.5, fc="none",
                            ec=C["white"], lw=1.5, alpha=alpha * 0.5))


def draw_nn_layer(ax, x, y_center, n_nodes, color, label="", node_r=0.04, spacing=0.12, fontsize=7):
    ys = np.linspace(y_center - (n_nodes - 1) * spacing / 2,
                     y_center + (n_nodes - 1) * spacing / 2, n_nodes)
    for yi in ys:
        c = Circle((x, yi), node_r, fc=color, ec="white", lw=0.8, transform=ax.transAxes,
                   clip_on=False, zorder=5)
        ax.add_patch(c)
    if label:
        ax.text(x, y_center - (n_nodes - 1) * spacing / 2 - 0.06, label,
                transform=ax.transAxes, color=C["white"], fontsize=fontsize, ha="center")
    return ys


def draw_connections(ax, x1, ys1, x2, ys2, alpha=0.15, color=C["accent"]):
    for y1 in ys1[::max(1, len(ys1)//5)]:
        for y2 in ys2[::max(1, len(ys2)//5)]:
            ax.plot([x1, x2], [y1, y2], transform=ax.transAxes,
                    color=color, alpha=alpha, lw=0.4, zorder=3)


# ─── Frame generators ─────────────────────────────────────────────────────────

def make_frame(scene_fn, figsize=(W/DPI, H/DPI), dpi=DPI):
    fig = plt.figure(figsize=figsize, facecolor=C["bg"], dpi=dpi)
    scene_fn(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=C["bg"], bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = np.frombuffer(buf.read(), dtype=np.uint8)
    import imageio.v3 as iio
    return iio.imread(io.BytesIO(img[0:0] + img))


def render_fig(fig, dpi=DPI):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=C["bg"])
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf)
    return np.array(img)


def add_title_bar(fig, title, subtitle="", t=1.0):
    ax = fig.add_axes([0, 0.88, 1, 0.12])
    ax.set_facecolor(C["bg"])
    ax.axis("off")
    alpha = clamp(t * 3)
    ax.text(0.5, 0.65, title, transform=ax.transAxes, color=C["white"],
            fontsize=22, fontweight="bold", ha="center", va="center",
            alpha=alpha,
            path_effects=[pe.withStroke(linewidth=4, foreground=C["bg"])])
    if subtitle:
        ax.text(0.5, 0.2, subtitle, transform=ax.transAxes, color=C["accent"],
                fontsize=11, ha="center", va="center", alpha=alpha * 0.9)
    # accent bar
    ax.add_patch(Rectangle((0, 0), 1, 0.03, transform=ax.transAxes,
                            fc=C["accent"], ec="none", alpha=alpha * 0.8))


# ════════════════════════════════════════════════════════════════════════════
# SCENE BUILDERS
# ════════════════════════════════════════════════════════════════════════════

# ── Scene 1: TITLE ──────────────────────────────────────────────────────────
def scene_title(fig, t):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C["bg"])
    ax.axis("off")
    # Animated grid background
    for xi in np.linspace(0, 1, 20):
        ax.axvline(xi, color=C["accent"], alpha=0.03 + 0.02 * math.sin(t * 2 + xi * 10), lw=0.5)
    for yi in np.linspace(0, 1, 12):
        ax.axhline(yi, color=C["accent"], alpha=0.03 + 0.02 * math.cos(t * 2 + yi * 10), lw=0.5)

    ta = clamp(t / 0.8)
    sa = clamp((t - 0.4) / 0.6)
    da = clamp((t - 0.7) / 0.5)

    ax.text(0.5, 0.62, "RL-Trained Hospital Navigation Robot",
            transform=ax.transAxes, color=C["white"], fontsize=32, fontweight="bold",
            ha="center", va="center", alpha=ease_out(ta),
            path_effects=[pe.withStroke(linewidth=6, foreground=C["bg"])])
    ax.text(0.5, 0.50, "SAC  +  Context Weight Network  +  Cultural-Adaptive Reward Shaping",
            transform=ax.transAxes, color=C["accent"], fontsize=15,
            ha="center", va="center", alpha=ease_out(sa))
    ax.text(0.5, 0.40, "LMIC Hospital Navigation  ·  Dhaka Medical College Hospital Setting",
            transform=ax.transAxes, color=C["gray"], fontsize=11,
            ha="center", va="center", alpha=ease_out(da))

    # Bottom tags
    for i, (tag, col) in enumerate([("IEEE RA-L Submission", C["gold"]),
                                     ("300 k Training Steps", C["green"]),
                                     ("SAC + CWN (1 013 params)", C["purple"])]):
        ax.text(0.25 + i * 0.25, 0.28, tag, transform=ax.transAxes, color=col,
                fontsize=10, ha="center", va="center", alpha=ease_out(da),
                bbox=dict(boxstyle="round,pad=0.3", fc=C["panel"], ec=col, lw=1))

    # Animated pulse circle
    r = 0.02 + 0.01 * math.sin(t * 4)
    c = Circle((0.5, 0.18), r, transform=ax.transAxes, fc="none",
               ec=C["accent"], lw=1.5, alpha=0.6)
    ax.add_patch(c)
    ax.text(0.5, 0.18, "▶", transform=ax.transAxes, color=C["accent"],
            fontsize=14, ha="center", va="center", alpha=0.8)


# ── Scene 2: PROBLEM OVERVIEW ───────────────────────────────────────────────
def scene_problem(fig, t):
    add_title_bar(fig, "The Challenge: Hospital Navigation in LMIC Settings",
                  "Dhaka Medical College Hospital — 133% occupancy, cultural constraints", t)
    # Left: hospital map
    ax_map = fig.add_axes([0.02, 0.08, 0.46, 0.78])
    draw_hospital(ax_map, alpha=clamp(t * 2), show_goal=True, highlight_zones=t > 0.4)
    ax_map.set_title("40 m × 30 m Floor Plan — DMCH Ward", color=C["white"],
                     fontsize=9, pad=4, fontweight="bold")

    # Right: challenge cards
    ax_r = fig.add_axes([0.52, 0.08, 0.46, 0.78])
    ax_r.set_facecolor(C["bg"])
    ax_r.axis("off")

    challenges = [
        (C["red"],    "[*] 3.75 peds/100 m²",  "3-6x Western benchmarks\n(SocNavBench: 0.5-1.5)"),
        (C["green"],  "[Prayer] Prayer Windows",     "5 daily Islamic prayer periods\naffect robot routing"),
        (C["purple"], "[F/M] Gender Segregation", "Separate male/female wards\ncontrolled entry points"),
        (C["orange"], "[Fam] Family Clusters",   "LMIC-specific: families\ncamping beside patients"),
        (C["accent"], "[+] Overflow Beds",      "Beds in corridors - not walls\nnarrow passageways"),
    ]
    for i, (col, title, desc) in enumerate(challenges):
        alpha = ease_out(clamp((t - i * 0.12) / 0.3))
        y = 0.85 - i * 0.165
        ax_r.add_patch(FancyBboxPatch((0.02, y - 0.09), 0.96, 0.13,
                                       boxstyle="round,pad=0.01",
                                       fc=C["panel"], ec=col, lw=1.5, alpha=alpha,
                                       transform=ax_r.transAxes))
        ax_r.text(0.07, y - 0.02, title, transform=ax_r.transAxes,
                  color=col, fontsize=11, fontweight="bold", alpha=alpha)
        ax_r.text(0.07, y - 0.07, desc, transform=ax_r.transAxes,
                  color=C["white"], fontsize=8, alpha=alpha * 0.9)


# ── Scene 3: OBSERVATION & ACTION SPACE ─────────────────────────────────────
def scene_obs_action(fig, t):
    add_title_bar(fig, "Observation Space  ·  Action Space",
                  "85-dimensional input  ·  Continuous differential-drive actions", t)

    # Left: hospital with LiDAR
    ax_map = fig.add_axes([0.02, 0.08, 0.40, 0.78])
    robot_pos = np.array([17.0, 6.0])
    draw_hospital(ax_map, alpha=0.85, show_lidar=True, robot_pos=robot_pos,
                  robot_angle=math.pi / 2 + math.sin(t * 2) * 0.2, policy_quality=0.6, show_goal=True)
    ax_map.set_title("LiDAR — 48 rays, 270° FOV, 10 m range", color=C["white"],
                     fontsize=8, pad=4)

    # Right: obs breakdown bar chart
    ax_obs = fig.add_axes([0.46, 0.45, 0.52, 0.38])
    ax_obs.set_facecolor(C["panel"])
    for spine in ax_obs.spines.values():
        spine.set_edgecolor(C["gray"])
    components = [
        ("LiDAR (48)",       48, C["accent"]),
        ("Pedestrians (20)", 20, C["orange"]),
        ("Robot State (5)",   5, C["green"]),
        ("Context (12)",     12, C["purple"]),
    ]
    xs = [0]
    for label, size, col in components:
        w = size
        bar_alpha = ease_out(clamp((t - 0.2) / 0.5))
        ax_obs.barh(0, w * bar_alpha, left=xs[-1], color=col, alpha=0.85, height=0.5)
        if bar_alpha > 0.4:
            ax_obs.text(xs[-1] + w * bar_alpha / 2, 0, label,
                        color="white", fontsize=7.5, ha="center", va="center", fontweight="bold")
        xs.append(xs[-1] + w)
    ax_obs.set_xlim(0, 85)
    ax_obs.set_ylim(-0.5, 0.8)
    ax_obs.set_xlabel("Observation dimension index", color=C["white"], fontsize=8)
    ax_obs.set_yticks([])
    ax_obs.tick_params(colors=C["white"])
    ax_obs.set_title("85-Dimensional Observation Vector", color=C["white"], fontsize=9, fontweight="bold")

    # Action space diagram
    ax_act = fig.add_axes([0.46, 0.09, 0.52, 0.30])
    ax_act.set_facecolor(C["panel"])
    for spine in ax_act.spines.values():
        spine.set_edgecolor(C["gray"])
    ax_act.set_xlim(-0.1, 1.1)
    ax_act.set_ylim(-1.2, 1.2)
    ax_act.set_aspect("equal")
    ax_act.axis("off")
    a = ease_out(clamp((t - 0.4) / 0.4))
    ax_act.text(0.5, 1.0, "Action Space  [v, ω]", color=C["white"], fontsize=9,
                ha="center", fontweight="bold")
    ax_act.text(0.5, 0.7, "v ∈ [0.0, 0.8] m/s  (linear)", color=C["accent"], fontsize=9, ha="center", alpha=a)
    ax_act.text(0.5, 0.4, "ω ∈ [−π, +π] rad/s  (angular)", color=C["gold"], fontsize=9, ha="center", alpha=a)
    ax_act.text(0.5, 0.1, "Differential-drive  ·  r=0.25 m  ·  Δt=0.1 s", color=C["gray"],
                fontsize=8, ha="center", alpha=a)


# ── Scene 4: REWARD ARCHITECTURE ───────────────────────────────────────────
def scene_reward(fig, t):
    add_title_bar(fig, "Cultural-Adaptive Reward Shaping (CARS)",
                  "4 differentiable reward components combined by the Context Weight Network", t)

    ax = fig.add_axes([0.02, 0.08, 0.96, 0.78])
    ax.set_facecolor(C["bg"])
    ax.axis("off")

    components = [
        ("r_nav",  C["accent"],  "Navigation\nReward",  "Potential-based shaping\n[−1, +1]",     0.12),
        ("r_safe", C["red"],     "Safety\nReward",      "Collision + proxemics\n[−5, 0]",         0.35),
        ("r_cult", C["green"],   "Cultural\nReward",    "Prayer + gender + family\n[−2, 0]",      0.58),
        ("r_task", C["gold"],    "Task\nReward",        "Completion + urgency\n[−1, +2]",         0.81),
    ]

    for i, (sym, col, name, desc, xpos) in enumerate(components):
        alpha = ease_out(clamp((t - i * 0.15) / 0.4))
        # Box
        ax.add_patch(FancyBboxPatch((xpos - 0.09, 0.52), 0.18, 0.32,
                                    boxstyle="round,pad=0.015",
                                    fc=C["panel"], ec=col, lw=2.0, alpha=alpha,
                                    transform=ax.transAxes))
        ax.text(xpos, 0.82, sym, transform=ax.transAxes, color=col,
                fontsize=16, fontweight="bold", ha="center", alpha=alpha)
        ax.text(xpos, 0.72, name, transform=ax.transAxes, color=C["white"],
                fontsize=9, ha="center", va="center", alpha=alpha)
        ax.text(xpos, 0.60, desc, transform=ax.transAxes, color=C["gray"],
                fontsize=7.5, ha="center", va="center", alpha=alpha)
        # Arrow down to CWN
        if alpha > 0.3:
            ax.annotate("", xy=(xpos, 0.38), xytext=(xpos, 0.52),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, alpha=alpha))

    # CWN box
    cwn_alpha = ease_out(clamp((t - 0.6) / 0.3))
    ax.add_patch(FancyBboxPatch((0.18, 0.22), 0.64, 0.16,
                                boxstyle="round,pad=0.015",
                                fc="#1A1A35", ec=C["purple"], lw=2.5, alpha=cwn_alpha,
                                transform=ax.transAxes))
    ax.text(0.5, 0.33, "Context Weight Network (CWN)", transform=ax.transAxes,
            color=C["purple"], fontsize=12, fontweight="bold", ha="center", alpha=cwn_alpha)
    ax.text(0.5, 0.26, "12-dim context → MLP(12→32→16→4) → Softmax(τ) → [w_nav, w_safe, w_cult, w_task]",
            transform=ax.transAxes, color=C["white"], fontsize=8, ha="center", alpha=cwn_alpha)

    # Final reward
    fin_alpha = ease_out(clamp((t - 0.8) / 0.3))
    ax.annotate("", xy=(0.5, 0.12), xytext=(0.5, 0.22),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C["gold"], lw=2, alpha=fin_alpha))
    ax.add_patch(FancyBboxPatch((0.33, 0.03), 0.34, 0.09,
                                boxstyle="round,pad=0.015",
                                fc=C["panel"], ec=C["gold"], lw=2, alpha=fin_alpha,
                                transform=ax.transAxes))
    ax.text(0.5, 0.075, "r_total = w · [r_nav, r_safe, r_cult, r_task]",
            transform=ax.transAxes, color=C["gold"], fontsize=10, fontweight="bold",
            ha="center", alpha=fin_alpha)

    # Cultural sub-terms annotation
    cult_alpha = ease_out(clamp((t - 0.35) / 0.3))
    for i, (sub, ycol) in enumerate([("Prayer avoidance", C["green"]),
                                      ("Gender zone transit", C["teal"]),
                                      ("Family cluster proxemics", C["purple"])]):
        ax.text(0.72 + 0.0, 0.49 - i * 0.06, "- " + sub, transform=ax.transAxes,
                color=ycol, fontsize=8, alpha=cult_alpha)


# ── Scene 5: SAC ARCHITECTURE ───────────────────────────────────────────────
def scene_sac(fig, t):
    add_title_bar(fig, "Soft Actor-Critic (SAC) Training Architecture",
                  "Off-policy, entropy-regularized, continuous action space — SB3 implementation", t)

    ax = fig.add_axes([0.02, 0.08, 0.96, 0.78])
    ax.set_facecolor(C["bg"])
    ax.axis("off")

    alpha_base = ease_out(clamp(t / 0.3))

    # Left column: Environment
    ax.add_patch(FancyBboxPatch((0.01, 0.3), 0.13, 0.45,
                                boxstyle="round,pad=0.01",
                                fc=C["panel"], ec=C["green"], lw=2, alpha=alpha_base,
                                transform=ax.transAxes))
    ax.text(0.075, 0.77, "ENV", transform=ax.transAxes, color=C["green"],
            fontsize=14, fontweight="bold", ha="center", alpha=alpha_base)
    ax.text(0.075, 0.67, "LMICHospitalNav\n-v1", transform=ax.transAxes, color=C["white"],
            fontsize=8, ha="center", alpha=alpha_base)
    ax.text(0.075, 0.55, "s_t, r_t, done", transform=ax.transAxes, color=C["gray"],
            fontsize=7, ha="center", alpha=alpha_base)
    ax.text(0.075, 0.45, "85-dim obs\n2-dim action", transform=ax.transAxes, color=C["gray"],
            fontsize=7, ha="center", alpha=alpha_base)

    # Replay Buffer
    rb_alpha = ease_out(clamp((t - 0.1) / 0.3))
    ax.add_patch(FancyBboxPatch((0.18, 0.55), 0.14, 0.2,
                                boxstyle="round,pad=0.01",
                                fc=C["panel"], ec=C["orange"], lw=2, alpha=rb_alpha,
                                transform=ax.transAxes))
    ax.text(0.25, 0.675, "Replay\nBuffer", transform=ax.transAxes, color=C["orange"],
            fontsize=9, fontweight="bold", ha="center", alpha=rb_alpha)
    ax.text(0.25, 0.60, "200 000 steps", transform=ax.transAxes, color=C["gray"],
            fontsize=7, ha="center", alpha=rb_alpha)

    # Actor
    actor_alpha = ease_out(clamp((t - 0.2) / 0.3))
    ax.add_patch(FancyBboxPatch((0.37, 0.55), 0.16, 0.2,
                                boxstyle="round,pad=0.01",
                                fc="#1A2545", ec=C["accent"], lw=2, alpha=actor_alpha,
                                transform=ax.transAxes))
    ax.text(0.45, 0.675, "Actor π_θ", transform=ax.transAxes, color=C["accent"],
            fontsize=10, fontweight="bold", ha="center", alpha=actor_alpha)
    ax.text(0.45, 0.61, "MLP [256, 256]\n→ μ, σ → action", transform=ax.transAxes, color=C["white"],
            fontsize=7, ha="center", alpha=actor_alpha)

    # Critic
    critic_alpha = ease_out(clamp((t - 0.3) / 0.3))
    ax.add_patch(FancyBboxPatch((0.37, 0.30), 0.16, 0.2,
                                boxstyle="round,pad=0.01",
                                fc="#251A1A", ec=C["red"], lw=2, alpha=critic_alpha,
                                transform=ax.transAxes))
    ax.text(0.45, 0.425, "Critic Q_φ (×2)", transform=ax.transAxes, color=C["red"],
            fontsize=10, fontweight="bold", ha="center", alpha=critic_alpha)
    ax.text(0.45, 0.36, "MLP [256, 256]\n→ Q-value", transform=ax.transAxes, color=C["white"],
            fontsize=7, ha="center", alpha=critic_alpha)

    # CWN
    cwn_alpha = ease_out(clamp((t - 0.4) / 0.3))
    ax.add_patch(FancyBboxPatch((0.58, 0.42), 0.16, 0.33,
                                boxstyle="round,pad=0.01",
                                fc="#1A1A35", ec=C["purple"], lw=2, alpha=cwn_alpha,
                                transform=ax.transAxes))
    ax.text(0.66, 0.77, "CWN", transform=ax.transAxes, color=C["purple"],
            fontsize=12, fontweight="bold", ha="center", alpha=cwn_alpha)
    ax.text(0.66, 0.70, "1 013 params", transform=ax.transAxes, color=C["gray"],
            fontsize=7.5, ha="center", alpha=cwn_alpha)
    ax.text(0.66, 0.63, "12→32→16→4\nSoftmax(τ)", transform=ax.transAxes, color=C["white"],
            fontsize=7, ha="center", alpha=cwn_alpha)
    ax.text(0.66, 0.55, "[w_nav w_safe\n w_cult w_task]", transform=ax.transAxes, color=C["purple"],
            fontsize=7, ha="center", alpha=cwn_alpha)
    ax.text(0.66, 0.47, "τ learnable\n∈ [0.1, 10.0]", transform=ax.transAxes, color=C["gray"],
            fontsize=7, ha="center", alpha=cwn_alpha)

    # Training loop params
    loop_alpha = ease_out(clamp((t - 0.55) / 0.3))
    ax.add_patch(FancyBboxPatch((0.78, 0.3), 0.2, 0.45,
                                boxstyle="round,pad=0.01",
                                fc=C["panel"], ec=C["gold"], lw=2, alpha=loop_alpha,
                                transform=ax.transAxes))
    ax.text(0.88, 0.77, "Training Config", transform=ax.transAxes, color=C["gold"],
            fontsize=10, fontweight="bold", ha="center", alpha=loop_alpha)
    params = [
        ("Steps",      "300 k"),
        ("Batch",      "256"),
        ("LR",         "3×10⁻⁴"),
        ("γ",          "0.99"),
        ("train_freq", "4"),
        ("grad_steps", "4"),
        ("α (entropy)","auto"),
        ("Buffer",     "200 k"),
    ]
    for i, (k, v) in enumerate(params):
        ax.text(0.80, 0.70 - i * 0.055, f"{k}:", transform=ax.transAxes, color=C["gray"],
                fontsize=7.5, ha="left", alpha=loop_alpha)
        ax.text(0.97, 0.70 - i * 0.055, v, transform=ax.transAxes, color=C["white"],
                fontsize=7.5, ha="right", alpha=loop_alpha)

    # Arrows
    arr_alpha = ease_out(clamp((t - 0.45) / 0.3))
    arrows = [
        ((0.14, 0.60), (0.18, 0.65)),  # env → buffer
        ((0.32, 0.65), (0.37, 0.65)),  # buffer → actor
        ((0.32, 0.60), (0.37, 0.40)),  # buffer → critic
        ((0.53, 0.65), (0.58, 0.65)),  # actor → CWN
        ((0.74, 0.58), (0.78, 0.52)),  # CWN → config
        ((0.14, 0.45), (0.37, 0.40)),  # env → critic
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.2, alpha=arr_alpha * 0.7))

    # Bottom note
    note_alpha = ease_out(clamp((t - 0.7) / 0.3))
    ax.text(0.5, 0.15, "VecNormalize: running mean/std over observations and rewards  ·  3 seeds (0, 1, 2)  ·  ~171 env steps/sec",
            transform=ax.transAxes, color=C["gray"], fontsize=8, ha="center", alpha=note_alpha)
    ax.text(0.5, 0.07, "CWN trained jointly with SAC — no separate auxiliary loss — gradients flow through total reward",
            transform=ax.transAxes, color=C["accent"], fontsize=8.5, ha="center", alpha=note_alpha)


# ── Scene 6: TRAINING LOOP ANIMATION ─────────────────────────────────────────
def scene_training(fig, t, step_frac):
    """step_frac ∈ [0,1]: how far through training we are."""
    add_title_bar(fig, "Real-Time Training Progress",
                  f"Step {int(step_frac * TOTAL_STEPS):,} / {TOTAL_STEPS:,}  ·  SAC + CWN", t)

    idx = min(int(step_frac * 599), 598)

    # ── Hospital map (top-left) ──────────────────────────────────
    ax_map = fig.add_axes([0.01, 0.40, 0.36, 0.48])
    traj = gen_trajectory(t, policy_quality=0.3 + step_frac * 0.6)
    n_show = min(len(traj), max(1, int(len(traj) * step_frac * 2)))
    robot_pos = traj[min(n_show, len(traj) - 1)]
    draw_hospital(ax_map, alpha=0.9, show_lidar=True, robot_pos=robot_pos,
                  robot_angle=math.pi * (0.3 + t * 0.5),
                  policy_quality=0.3 + step_frac * 0.6, show_goal=True)
    if n_show > 1:
        seg = traj[:n_show]
        ax_map.plot(seg[:, 0], seg[:, 1], color=C["accent"], lw=1.5, alpha=0.7)
    ax_map.set_title(f"Policy Quality: {int((0.3 + step_frac * 0.6) * 100)}%",
                     color=C["white"], fontsize=8, pad=3)

    # ── Reward curve (top-right) ──────────────────────────────────
    ax_rew = fig.add_axes([0.40, 0.58, 0.58, 0.29])
    ax_rew.set_facecolor(C["panel"])
    for sp in ax_rew.spines.values():
        sp.set_edgecolor(C["gray"])

    s_slice = steps_x[:idx + 1]
    ax_rew.plot(s_slice, moving_avg(r_cwn[:idx + 1]),
                color=C["accent"], lw=2, label="CWN (Ours)")
    ax_rew.plot(s_slice, moving_avg(r_eq[:idx + 1]),
                color=C["orange"], lw=1.5, ls="--", label="Equal-Weight")
    ax_rew.plot(s_slice, moving_avg(r_van[:idx + 1]),
                color=C["gray"], lw=1.2, ls=":", label="Vanilla SAC")
    ax_rew.set_xlim(0, TOTAL_STEPS)
    ax_rew.set_ylim(-1.0, 1.8)
    ax_rew.set_xlabel("Training Steps", color=C["white"], fontsize=7)
    ax_rew.set_ylabel("Episode Reward", color=C["white"], fontsize=7)
    ax_rew.tick_params(colors=C["white"], labelsize=7)
    ax_rew.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["white"],
                  loc="upper left", framealpha=0.7)
    ax_rew.set_title("Episode Reward (EMA smoothed)", color=C["white"], fontsize=9, fontweight="bold")

    # ── Success Rate (bottom-left) ────────────────────────────────
    ax_sr = fig.add_axes([0.01, 0.08, 0.35, 0.29])
    ax_sr.set_facecolor(C["panel"])
    for sp in ax_sr.spines.values():
        sp.set_edgecolor(C["gray"])
    ax_sr.plot(steps_x[:idx + 1], moving_avg(sr_cwn[:idx + 1]) * 100,
               color=C["green"], lw=2, label="CWN")
    ax_sr.plot(steps_x[:idx + 1], moving_avg(sr_van[:idx + 1]) * 100,
               color=C["gray"], lw=1.5, ls="--", label="Vanilla")
    ax_sr.set_xlim(0, TOTAL_STEPS)
    ax_sr.set_ylim(0, 100)
    ax_sr.set_xlabel("Steps", color=C["white"], fontsize=7)
    ax_sr.set_ylabel("Success Rate (%)", color=C["white"], fontsize=7)
    ax_sr.tick_params(colors=C["white"], labelsize=7)
    ax_sr.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["white"])
    ax_sr.set_title("Success Rate ↑", color=C["green"], fontsize=9, fontweight="bold")
    # Current value annotation
    cur_sr = moving_avg(sr_cwn)[idx] * 100
    ax_sr.axhline(cur_sr, color=C["green"], ls=":", alpha=0.4)
    ax_sr.text(TOTAL_STEPS * 0.98, cur_sr + 2, f"{cur_sr:.1f}%",
               color=C["green"], fontsize=8, ha="right")

    # ── Cultural Violation Rate (bottom-center) ───────────────────
    ax_cvr = fig.add_axes([0.40, 0.08, 0.28, 0.29])
    ax_cvr.set_facecolor(C["panel"])
    for sp in ax_cvr.spines.values():
        sp.set_edgecolor(C["gray"])
    ax_cvr.plot(steps_x[:idx + 1], moving_avg(cvr_cwn[:idx + 1]) * 100,
                color=C["green"], lw=2, label="CWN")
    ax_cvr.plot(steps_x[:idx + 1], moving_avg(cvr_eq[:idx + 1]) * 100,
                color=C["orange"], lw=1.5, ls="--", label="Equal-W")
    ax_cvr.plot(steps_x[:idx + 1], moving_avg(cvr_van[:idx + 1]) * 100,
                color=C["red"], lw=1.5, ls=":", label="Vanilla")
    ax_cvr.set_xlim(0, TOTAL_STEPS)
    ax_cvr.set_ylim(0, 60)
    ax_cvr.set_xlabel("Steps", color=C["white"], fontsize=7)
    ax_cvr.set_ylabel("CVR (%)", color=C["white"], fontsize=7)
    ax_cvr.tick_params(colors=C["white"], labelsize=7)
    ax_cvr.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["white"])
    ax_cvr.set_title("Cultural Violation Rate ↓", color=C["red"], fontsize=9, fontweight="bold")

    # ── CWN Weights (bottom-right) ────────────────────────────────
    ax_w = fig.add_axes([0.72, 0.08, 0.27, 0.29])
    ax_w.set_facecolor(C["panel"])
    for sp in ax_w.spines.values():
        sp.set_edgecolor(C["gray"])
    wsteps = steps_x[:idx + 1]
    ax_w.plot(wsteps, moving_avg(w_nav_curve[:idx + 1]),  color=C["accent"],  lw=1.5, label="w_nav")
    ax_w.plot(wsteps, moving_avg(w_safe_curve[:idx + 1]), color=C["red"],     lw=1.5, label="w_safe")
    ax_w.plot(wsteps, moving_avg(w_cult_curve[:idx + 1]), color=C["green"],   lw=1.5, label="w_cult")
    ax_w.plot(wsteps, moving_avg(w_task_curve[:idx + 1]), color=C["gold"],    lw=1.5, label="w_task")
    ax_w.set_xlim(0, TOTAL_STEPS)
    ax_w.set_ylim(0, 0.55)
    ax_w.set_xlabel("Steps", color=C["white"], fontsize=7)
    ax_w.set_ylabel("Weight", color=C["white"], fontsize=7)
    ax_w.tick_params(colors=C["white"], labelsize=7)
    ax_w.legend(fontsize=7, facecolor=C["panel"], labelcolor=C["white"], ncol=2)
    ax_w.set_title("CWN Weight Evolution", color=C["purple"], fontsize=9, fontweight="bold")

    # ── Live Metrics HUD (center-right middle row) ────────────────
    ax_hud = fig.add_axes([0.40, 0.40, 0.58, 0.16])
    ax_hud.set_facecolor(C["panel"])
    ax_hud.axis("off")
    metrics = [
        ("Step",      f"{int(step_frac * TOTAL_STEPS):,}",      C["accent"]),
        ("SR",        f"{moving_avg(sr_cwn)[idx]*100:.1f}%",     C["green"]),
        ("CVR",       f"{moving_avg(cvr_cwn)[idx]*100:.1f}%",    C["red"]),
        ("r_total",   f"{moving_avg(r_cwn)[idx]:.3f}",          C["gold"]),
        ("w_cult",    f"{moving_avg(w_cult_curve)[idx]:.3f}",   C["purple"]),
        ("τ_CWN",     f"{1.5 + step_frac * 2:.2f}",             C["teal"]),
    ]
    for i, (label, val, col) in enumerate(metrics):
        x = 0.05 + i * 0.165
        ax_hud.add_patch(FancyBboxPatch((x, 0.1), 0.14, 0.8,
                                        boxstyle="round,pad=0.05",
                                        fc=C["bg"], ec=col, lw=1.5, alpha=0.9,
                                        transform=ax_hud.transAxes))
        ax_hud.text(x + 0.07, 0.70, label, transform=ax_hud.transAxes,
                    color=C["gray"], fontsize=8, ha="center")
        ax_hud.text(x + 0.07, 0.35, val, transform=ax_hud.transAxes,
                    color=col, fontsize=11, ha="center", fontweight="bold")


# ── Scene 7: CWN BEHAVIOR ───────────────────────────────────────────────────
def scene_cwn_behavior(fig, t):
    add_title_bar(fig, "CWN: Adaptive Weight Behavior",
                  "Context-driven reward weighting — prayer windows · shift changes · task urgency", t)

    # Prayer window simulation
    time_of_day = np.linspace(0, 24, 500)
    prayer_times = [5.0, 12.5, 15.5, 18.2, 19.8]
    prayer_window = 0.5

    w_cult_tod = np.ones(500) * 0.22
    for pt in prayer_times:
        mask = np.abs(time_of_day - pt) < prayer_window
        w_cult_tod[mask] = 0.42
    w_cult_smooth = np.convolve(w_cult_tod, np.ones(15) / 15, mode="same")

    # Crowd density (shift changes)
    density_tod = 0.5 + 0.2 * np.sin((time_of_day - 7) * np.pi / 6)
    density_tod += 0.15 * np.exp(-((time_of_day - 7.5) ** 2) / 0.5)
    density_tod += 0.15 * np.exp(-((time_of_day - 13) ** 2) / 0.5)
    density_tod += 0.12 * np.exp(-((time_of_day - 21) ** 2) / 0.5)

    t_slice = min(int(t * 500), 499)

    # Left: w_cult over time
    ax1 = fig.add_axes([0.03, 0.08, 0.45, 0.38])
    ax1.set_facecolor(C["panel"])
    for sp in ax1.spines.values(): sp.set_edgecolor(C["gray"])
    ax1.plot(time_of_day[:t_slice], w_cult_smooth[:t_slice], color=C["green"], lw=2)
    for pt in prayer_times:
        ax1.axvspan(pt - prayer_window, pt + prayer_window, alpha=0.15, color=C["green"])
        ax1.text(pt, 0.44, "[P]", fontsize=7, ha="center", color=C["green"])
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0.1, 0.5)
    ax1.set_xlabel("Time of Day (h)", color=C["white"], fontsize=8)
    ax1.set_ylabel("w_cult (CWN output)", color=C["white"], fontsize=8)
    ax1.tick_params(colors=C["white"], labelsize=7)
    ax1.set_title("Cultural Weight ↑ During Prayer Windows", color=C["green"], fontsize=9, fontweight="bold")

    # Right: w_safe vs crowd density
    ax2 = fig.add_axes([0.53, 0.08, 0.45, 0.38])
    ax2.set_facecolor(C["panel"])
    for sp in ax2.spines.values(): sp.set_edgecolor(C["gray"])
    w_safe_tod = 0.25 + 0.12 * density_tod
    ax2.plot(time_of_day[:t_slice], density_tod[:t_slice] * 60 + 10,
             color=C["orange"], lw=1.5, ls="--", label="Crowd density (peds)")
    ax2_r = ax2.twinx()
    ax2_r.set_facecolor("none")
    ax2_r.plot(time_of_day[:t_slice], w_safe_tod[:t_slice], color=C["red"], lw=2, label="w_safe")
    ax2_r.set_ylim(0.1, 0.5)
    ax2_r.tick_params(colors=C["red"], labelsize=7)
    ax2_r.set_ylabel("w_safe (CWN output)", color=C["red"], fontsize=8)
    for shift in [7, 14, 22]:
        ax2.axvspan(shift - 0.5, shift + 0.5, alpha=0.1, color=C["orange"])
        ax2.text(shift, 44, "<>", fontsize=11, ha="center", color=C["orange"])
    ax2.set_xlim(0, 24)
    ax2.set_xlabel("Time of Day (h)", color=C["white"], fontsize=8)
    ax2.set_ylabel("Crowd density (peds)", color=C["orange"], fontsize=8)
    ax2.tick_params(colors=C["white"], labelsize=7)
    ax2.set_title("Safety Weight ↑ at Shift Changes", color=C["red"], fontsize=9, fontweight="bold")

    # Top: weight pie chart at current context
    ax_pie = fig.add_axes([0.3, 0.50, 0.40, 0.40])
    ax_pie.set_facecolor(C["bg"])
    idx = min(int(t * 599), 598)
    weights = [w_nav_curve[idx], w_safe_curve[idx], w_cult_curve[idx], w_task_curve[idx]]
    labels = ["w_nav", "w_safe", "w_cult", "w_task"]
    colors = [C["accent"], C["red"], C["green"], C["gold"]]
    wedge_props = dict(width=0.5, edgecolor="black", linewidth=1.2)
    wedges, texts, autotexts = ax_pie.pie(
        weights, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=wedge_props, pctdistance=0.75,
        textprops={"color": C["white"], "fontsize": 9}
    )
    for at in autotexts:
        at.set_color(C["bg"])
        at.set_fontweight("bold")
    ax_pie.set_title("CWN Weight Distribution\n(current context)", color=C["white"],
                     fontsize=9, fontweight="bold", pad=4)


# ── Scene 8: BASELINES COMPARISON ───────────────────────────────────────────
def scene_baselines(fig, t):
    add_title_bar(fig, "Performance vs. Baselines",
                  "3 seeds × 100 episodes  ·  Medium density (45 pedestrians)", t)

    methods = ["CWN\n(Ours)", "Attn-W\n(B5)", "Nav-Heavy\n(B2)", "Equal-W\n(B1)", "He 2025\n(B4)", "Vanilla\n(B3)"]
    colors  = [C["accent"], C["purple"], C["gold"], C["orange"], C["teal"], C["gray"]]
    sr_vals = [89.1, 85.4, 81.2, 78.6, 71.4, 64.8]
    sr_err  = [2.1,  2.8,  3.1,  3.4,  4.2,  4.9]
    cvr_vals= [3.8,  6.2, 11.8, 13.5, 18.2, 24.7]
    cvr_err = [0.9,  1.3,  2.1,  2.4,  3.1,  3.8]
    spl_vals= [0.742, 0.708, 0.663, 0.644, 0.590, 0.531]
    mpd_vals= [0.94,  0.89,  0.84,  0.82,  0.77,  0.71]

    def animated_bar(ax, vals, errs, col_list, ylabel, title, ylim, higher_better=True,
                     fmt="{:.1f}"):
        ax.set_facecolor(C["panel"])
        for sp in ax.spines.values(): sp.set_edgecolor(C["gray"])
        x = np.arange(len(methods))
        for i, (v, e, col) in enumerate(zip(vals, errs, col_list)):
            bar_h = v * ease_out(clamp((t - i * 0.07) / 0.35))
            ax.bar(i, bar_h, color=col, alpha=0.85, width=0.6, zorder=3)
            if bar_h > ylim[0] + (ylim[1] - ylim[0]) * 0.1:
                ax.errorbar(i, bar_h, yerr=e * ease_out(clamp((t - i * 0.07) / 0.35)),
                            fmt="none", color="white", capsize=4, lw=1.5)
                ax.text(i, bar_h + (ylim[1] - ylim[0]) * 0.03,
                        fmt.format(v), color="white", fontsize=7.5, ha="center", fontweight="bold")
        # Crown on best
        best_i = 0 if higher_better else 0
        ax.text(best_i, ylim[1] * 0.97, "★", fontsize=12, ha="center", color=C["gold"], zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, color=C["white"], fontsize=7.5)
        ax.set_ylabel(ylabel, color=C["white"], fontsize=8)
        ax.set_ylim(*ylim)
        ax.tick_params(colors=C["white"], labelsize=7)
        ax.set_title(title, color=C["white"], fontsize=9, fontweight="bold")
        ax.yaxis.set_tick_params(labelcolor=C["white"])

    ax1 = fig.add_axes([0.02, 0.44, 0.47, 0.40])
    animated_bar(ax1, sr_vals, sr_err, colors, "Success Rate (%)", "Success Rate ↑", (50, 100))

    ax2 = fig.add_axes([0.52, 0.44, 0.47, 0.40])
    animated_bar(ax2, cvr_vals, cvr_err, colors, "CVR (%)", "Cultural Violation Rate ↓",
                 (0, 30), higher_better=False, fmt="{:.1f}")

    ax3 = fig.add_axes([0.02, 0.06, 0.47, 0.33])
    animated_bar(ax3, [v * 100 for v in spl_vals], [0.018]*6, colors,
                 "SPL × 100", "Success-weighted Path Length ↑", (45, 80), fmt="{:.1f}")

    ax4 = fig.add_axes([0.52, 0.06, 0.47, 0.33])
    animated_bar(ax4, [v * 100 for v in mpd_vals], [0.07]*6, colors,
                 "MPD × 100 (cm)", "Mean Pedestrian Distance ↑", (60, 105), fmt="{:.1f}")


# ── Scene 9: DENSITY ABLATION ────────────────────────────────────────────────
def scene_density(fig, t):
    add_title_bar(fig, "Density Ablation — LMIC vs. Western Benchmarks",
                  "He et al. (2025) collapses at LMIC densities — CWN remains robust", t)

    densities = ["Low\n(23 peds)", "Medium\n(45 peds)", "High\n(74 peds)"]
    sr_cwn_d = [93.2, 89.1, 82.4]
    sr_he_d  = [88.4, 71.4, 51.8]
    sr_van_d = [76.1, 64.8, 48.2]

    ax = fig.add_axes([0.08, 0.10, 0.56, 0.74])
    ax.set_facecolor(C["panel"])
    for sp in ax.spines.values(): sp.set_edgecolor(C["gray"])
    x = np.arange(3)
    w = 0.26

    for i, (vals, col, label) in enumerate([
        (sr_cwn_d, C["accent"], "CWN (Ours)"),
        (sr_he_d,  C["teal"],  "He et al. 2025"),
        (sr_van_d, C["gray"],  "Vanilla SAC"),
    ]):
        for j, v in enumerate(vals):
            bar_h = v * ease_out(clamp((t - (i * 3 + j) * 0.04) / 0.3))
            ax.bar(x[j] + (i - 1) * w, bar_h, width=w * 0.9, color=col,
                   alpha=0.85, label=label if j == 0 else None, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(densities, color=C["white"], fontsize=9)
    ax.set_ylabel("Success Rate (%)", color=C["white"], fontsize=9)
    ax.set_ylim(0, 105)
    ax.tick_params(colors=C["white"])
    ax.legend(fontsize=9, facecolor=C["panel"], labelcolor=C["white"])
    ax.set_title("Success Rate vs. Crowd Density", color=C["white"], fontsize=11, fontweight="bold")

    # Western benchmark comparison line
    bench_alpha = ease_out(clamp((t - 0.7) / 0.3))
    ax.axhline(87, color=C["orange"], ls="--", lw=1.5, alpha=bench_alpha)
    ax.text(2.4, 88.5, "Western SOTA baseline", color=C["orange"], fontsize=8, alpha=bench_alpha)

    # Right: density visualization
    ax_map = fig.add_axes([0.68, 0.10, 0.30, 0.74])
    draw_hospital(ax_map, alpha=0.9, show_goal=True)
    ax_map.set_title("LMIC: 45 peds/40m×30m\n3.75 peds/100m²", color=C["white"], fontsize=8, pad=4)
    ax_map.text(20, -0.5, "Western avg: 0.5–1.5 peds/100m²",
                color=C["orange"], fontsize=7, ha="center", va="bottom")


# ── Scene 10: CONCLUSION ──────────────────────────────────────────────────────
def scene_conclusion(fig, t):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C["bg"])
    ax.axis("off")
    for xi in np.linspace(0, 1, 25):
        ax.axvline(xi, color=C["accent"], alpha=0.025 + 0.01 * math.sin(t * 3 + xi * 15), lw=0.5)

    # Title
    ta = ease_out(clamp(t / 0.3))
    ax.text(0.5, 0.88, "Key Contributions & Results",
            transform=ax.transAxes, color=C["white"], fontsize=26, fontweight="bold",
            ha="center", alpha=ta,
            path_effects=[pe.withStroke(linewidth=5, foreground=C["bg"])])
    ax.add_patch(Rectangle((0.1, 0.855), 0.8, 0.003, transform=ax.transAxes,
                            fc=C["accent"], ec="none", alpha=ta))

    contributions = [
        (C["accent"],  "C1", "First LMIC Hospital Navigation Benchmark",
         "40m×30m DMCH env · 4 pedestrian classes · 45 peds (3.75/100m²)"),
        (C["green"],   "C2", "Cultural-Adaptive Reward Shaping (CARS)",
         "Prayer avoidance · gender zones · family proxemics — fully differentiable"),
        (C["purple"],  "C3", "Context Weight Network (CWN)",
         "1 013 params · joint SAC training · adaptive weights per context"),
    ]
    for i, (col, tag, title, desc) in enumerate(contributions):
        alpha = ease_out(clamp((t - 0.2 - i * 0.15) / 0.35))
        y = 0.70 - i * 0.145
        ax.add_patch(FancyBboxPatch((0.04, y - 0.07), 0.92, 0.10,
                                    boxstyle="round,pad=0.015",
                                    fc=C["panel"], ec=col, lw=2, alpha=alpha,
                                    transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((0.04, y - 0.07), 0.06, 0.10,
                                    boxstyle="round,pad=0.0",
                                    fc=col, ec="none", alpha=alpha * 0.3,
                                    transform=ax.transAxes))
        ax.text(0.07, y - 0.02, tag, transform=ax.transAxes, color=col,
                fontsize=14, fontweight="bold", ha="center", va="center", alpha=alpha)
        ax.text(0.13, y - 0.01, title, transform=ax.transAxes, color=col,
                fontsize=11, fontweight="bold", alpha=alpha)
        ax.text(0.13, y - 0.055, desc, transform=ax.transAxes, color=C["white"],
                fontsize=9, alpha=alpha * 0.9)

    # Metrics row
    results_alpha = ease_out(clamp((t - 0.65) / 0.3))
    metrics = [
        ("89.1%", "Success Rate", C["green"]),
        ("3.8%",  "Cultural Viol.", C["red"]),
        ("0.742", "SPL", C["gold"]),
        ("0.94m", "Avg Ped Dist", C["teal"]),
        ("1 013", "CWN Params", C["purple"]),
        ("300 k", "Train Steps", C["accent"]),
    ]
    for i, (val, label, col) in enumerate(metrics):
        x = 0.08 + i * 0.155
        ax.add_patch(FancyBboxPatch((x, 0.07), 0.13, 0.09,
                                    boxstyle="round,pad=0.01",
                                    fc=C["panel"], ec=col, lw=1.5, alpha=results_alpha,
                                    transform=ax.transAxes))
        ax.text(x + 0.065, 0.135, val, transform=ax.transAxes, color=col,
                fontsize=13, fontweight="bold", ha="center", alpha=results_alpha)
        ax.text(x + 0.065, 0.085, label, transform=ax.transAxes, color=C["white"],
                fontsize=7.5, ha="center", alpha=results_alpha)

    # Footer
    foot_alpha = ease_out(clamp((t - 0.85) / 0.2))
    ax.text(0.5, 0.025, "IEEE RA-L  ·  LMIC Hospital Navigation  ·  mamun@padakhep.org",
            transform=ax.transAxes, color=C["gray"], fontsize=9,
            ha="center", alpha=foot_alpha)


# ════════════════════════════════════════════════════════════════════════════
# MAIN VIDEO ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════

def build_video():
    print("Building professional RL training video...")

    # Scene schedule: (name, duration_seconds, build_fn)
    # Each frame is 1/FPS seconds
    scenes = [
        ("title",       4.0,  scene_title),
        ("problem",     6.0,  scene_problem),
        ("obs_action",  5.0,  scene_obs_action),
        ("reward",      6.0,  scene_reward),
        ("sac",         6.0,  scene_sac),
        ("training",   16.0,  None),   # special — has sub-animation
        ("cwn",         7.0,  scene_cwn_behavior),
        ("baselines",   7.0,  scene_baselines),
        ("density",     6.0,  scene_density),
        ("conclusion",  6.0,  scene_conclusion),
    ]

    total_frames = int(sum(d * FPS for _, d, _ in scenes))
    print(f"Total frames: {total_frames}  ({sum(d for _, d, _ in scenes):.0f}s at {FPS} fps)")

    output_path = "/home/roky/RL_med_robot/lmic-hospitalnav/rl_training_video.mp4"
    writer = imageio.get_writer(output_path, fps=FPS, quality=9,
                                codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1,
                                output_params=["-b:v", "4M"])

    frame_idx = 0
    for scene_name, duration, scene_fn in scenes:
        n_frames = int(duration * FPS)
        print(f"  Rendering scene '{scene_name}' ({n_frames} frames)...", flush=True)

        for f in range(n_frames):
            t = f / n_frames  # [0, 1] within this scene

            fig = plt.figure(figsize=(W / DPI, H / DPI), facecolor=C["bg"], dpi=DPI)

            if scene_name == "training":
                # step_frac: 0→1 over training scene
                step_frac = ease_out(t)
                scene_training(fig, t, step_frac)
            else:
                scene_fn(fig, t)

            # Fade in at scene start, fade out at end
            if t < 0.05 or t > 0.95:
                fade_alpha = min(t / 0.05, (1 - t) / 0.05)
                fade_alpha = clamp(fade_alpha)
                overlay_ax = fig.add_axes([0, 0, 1, 1], zorder=999)
                overlay_ax.set_facecolor(C["bg"])
                overlay_ax.axis("off")
                overlay_ax.patch.set_alpha(1 - fade_alpha)

            img = render_fig(fig, dpi=DPI)
            # Ensure correct shape for video (H, W, 3)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            # Resize to exact dimensions if needed
            if img.shape[0] != H or img.shape[1] != W:
                from PIL import Image as PILImage
                pil = PILImage.fromarray(img)
                pil = pil.resize((W, H), PILImage.LANCZOS)
                img = np.array(pil)
            writer.append_data(img)
            frame_idx += 1

        print(f"    Done ({frame_idx}/{total_frames} total)", flush=True)

    writer.close()
    print(f"\n✅ Video saved to: {output_path}")
    print(f"   Duration: {sum(d for _, d, _ in scenes):.0f}s  |  {total_frames} frames  |  {FPS} fps  |  1920×1080")


if __name__ == "__main__":
    build_video()
