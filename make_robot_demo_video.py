"""
1-Minute Public Demo Video: RL Robot Navigation in Hospital
Clean, cinematic — shows the robot learning over training episodes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import imageio
import math
import io
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────
FPS    = 30
DPI    = 100
VW, VH = 1920, 1080   # video dimensions

# ── Palette ───────────────────────────────────────────────────────────────
BG      = "#050B18"
PANEL   = "#0D1526"
WALL    = "#1A2340"
CORR    = "#101D35"
WARD_M  = "#0F1E3A"
WARD_F  = "#1A0F3A"
PRAYER  = "#0F2A1A"
ICU     = "#2A0F0F"
NURSE   = "#0F2A2A"
WHITE   = "#EDF2FF"
GRAY    = "#4A5568"
ACCENT  = "#00D4FF"   # cyan — robot / navigation
GREEN   = "#00FF88"   # success / goal
RED     = "#FF4466"   # collision / failure
YELLOW  = "#FFD700"   # reward / goal marker
PURPLE  = "#BB88FF"   # cultural zones
ORANGE  = "#FF8844"   # pedestrians
PINK    = "#FF44AA"   # family clusters

rng = np.random.default_rng(7)

# ── Hospital layout (simplified, in metres, 40×30 grid) ───────────────────
# Rooms list: (x, y, w, h, fill_color, border_color, label, label_color)
ROOMS = [
    (0,  15, 18, 7,  WARD_M,  "#2255AA", "MALE WARD",     "#5588FF"),
    (22, 15, 18, 7,  WARD_F,  "#882299", "FEMALE WARD",   PURPLE),
    (28, 22, 12, 8,  PRAYER,  "#00AA55", "PRAYER ROOM",   GREEN),
    (0,  22, 10, 8,  ICU,     "#AA2222", "ICU",           RED),
    (10, 22, 8,  8,  PANEL,   GRAY,      "SUPPLY",        GRAY),
    (0,  0,  8,  12, PANEL,   GRAY,      "PHARMACY",      GRAY),
    (32, 0,  8,  12, ICU,     "#AA2222", "EMERGENCY",     RED),
    (14, 0,  12, 12, NURSE,   ACCENT,    "NURSE STATION", ACCENT),
]
CORRIDOR = (0, 12, 40, 3)

# Overflow beds in corridor
OVERFLOW_BEDS = [(1 + i * 3.6, 12.2, 1.4, 0.8) for i in range(10)]

# ── Pedestrian snapshot ────────────────────────────────────────────────────
def make_pedestrians(n=45):
    peds = []
    positions_zones = [
        (8,  0, 15, 18,  7),   # male ward
        (10, 22, 15, 18,  7),  # female ward
        (12,  0, 12, 40,  3),  # corridor
        (8,   0, 22, 28,  8),  # icu + supply
        (7,   0,  0, 40, 12),  # lower level
    ]
    for count, zx, zy, zw, zh in positions_zones:
        for _ in range(count):
            peds.append({
                "pos": np.array([rng.uniform(zx + 0.3, zx + zw - 0.3),
                                 rng.uniform(zy + 0.3, zy + zh - 0.3)]),
                "vel": rng.uniform(-0.05, 0.05, 2),
                "r":   rng.uniform(0.22, 0.35),
                "col": rng.choice([ORANGE, PURPLE, PINK, "#88AAFF"]),
            })
    return peds

PEDS_STATIC = make_pedestrians()

def jitter_peds(peds, t):
    """Slightly animate pedestrians."""
    out = []
    for i, p in enumerate(peds):
        angle = t * 0.4 + i * 0.7
        drift = np.array([math.sin(angle) * 0.15, math.cos(angle * 1.3) * 0.1])
        out.append({**p, "pos": p["pos"] + drift})
    return out

# ── Trajectory generators ──────────────────────────────────────────────────
START = np.array([20.0, 6.0])   # nurse station
GOAL  = np.array([ 4.0,18.5])  # male ward

def waypoint_path(waypoints, steps_per_seg=25, noise=0.0):
    """Smooth path through waypoints with optional noise."""
    path = [waypoints[0].copy()]
    for wp in waypoints[1:]:
        cur = path[-1].copy()
        for _ in range(steps_per_seg):
            to = wp - cur
            d = np.linalg.norm(to)
            if d < 0.1:
                break
            cur = cur + to / d * 0.55
            if noise > 0:
                cur += rng.normal(0, noise, 2)
            path.append(cur.copy())
    return np.array(path)

# ── Episode trajectories (pre-computed) ────────────────────────────────────
# Untrained: random walk, ends nowhere near goal
def traj_untrained():
    pts = [START.copy()]
    pos = START.copy()
    for _ in range(90):
        angle = rng.uniform(0, 2 * math.pi)
        step  = np.array([math.cos(angle), math.sin(angle)]) * 0.55
        pos   = pos + step
        pos   = np.clip(pos, [0.3, 0.3], [39.7, 29.7])
        pts.append(pos.copy())
    return np.array(pts)

# Partially trained: heads roughly toward goal but deviates
def traj_partial():
    wps = [START,
           np.array([20.0, 10.0]),
           np.array([14.0, 13.5]),
           np.array([10.0, 15.0]),
           np.array([ 7.0, 16.0]),
           np.array([ 4.5, 18.0])]
    return waypoint_path(wps, steps_per_seg=22, noise=0.35)

# Trained: clean path, navigates corridor, avoids prayer room
def traj_trained():
    wps = [START,
           np.array([20.0,  9.0]),
           np.array([16.0, 13.0]),
           np.array([10.0, 13.5]),
           np.array([ 6.0, 14.8]),
           np.array([ 4.5, 15.5]),
           GOAL]
    return waypoint_path(wps, steps_per_seg=20, noise=0.08)

# Cultural aware: trained + extra loop to show prayer room avoidance
def traj_cultural():
    wps = [START,
           np.array([20.0,  9.0]),
           np.array([16.0, 13.0]),
           np.array([10.0, 13.5]),
           np.array([ 6.0, 14.8]),
           np.array([ 4.5, 15.5]),
           GOAL]
    path = waypoint_path(wps, steps_per_seg=20, noise=0.04)
    # Hold at goal
    return np.vstack([path, np.tile(GOAL, (20, 1))])

TRAJS = {
    "untrained": traj_untrained(),
    "partial":   traj_partial(),
    "trained":   traj_trained(),
    "cultural":  traj_cultural(),
}

# ── Drawing helpers ────────────────────────────────────────────────────────
def draw_map(ax, show_zones=False, zone_pulse=0.0):
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, 40.5)
    ax.set_ylim(-0.5, 30.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Floor
    ax.add_patch(Rectangle((-0.5, -0.5), 41.5, 31.5,
                            fc=CORR, ec="none"))

    # Cultural zone glows (pulsing)
    if show_zones:
        pa = 0.08 + 0.06 * zone_pulse
        ax.add_patch(Rectangle((28, 22), 12, 8, fc=GREEN,   alpha=pa, ec="none"))
        ax.add_patch(Rectangle((0, 15),  18,  7, fc="#2255AA", alpha=pa * 0.7, ec="none"))
        ax.add_patch(Rectangle((22, 15), 18,  7, fc=PURPLE,  alpha=pa * 0.7, ec="none"))

    # Rooms
    for (rx, ry, rw, rh, rc, ec, lbl, lc) in ROOMS:
        ax.add_patch(Rectangle((rx, ry), rw, rh,
                               fc=rc, ec=ec, lw=1.2, alpha=0.95))
        ax.text(rx + rw / 2, ry + rh / 2, lbl,
                color=lc, fontsize=5.5, ha="center", va="center",
                fontweight="bold", alpha=0.85,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

    # Corridor label
    ax.text(20, 13.5, "MAIN CORRIDOR", color=GRAY, fontsize=5,
            ha="center", va="center", alpha=0.7)

    # Overflow beds
    for (bx, by, bw, bh) in OVERFLOW_BEDS:
        ax.add_patch(Rectangle((bx, by), bw, bh,
                               fc="#243050", ec=GRAY, lw=0.5, alpha=0.8))

    # Outer walls
    ax.add_patch(Rectangle((-0.5, -0.5), 41.5, 31.5,
                            fc="none", ec="#2A3A5A", lw=2.0))


def draw_pedestrians(ax, peds, alpha=1.0):
    for p in peds:
        ax.add_patch(Circle(p["pos"], p["r"],
                            fc=p["col"], ec="none", alpha=alpha * 0.75, zorder=4))


def draw_robot(ax, pos, angle, trail=None, color=ACCENT, trail_color=ACCENT,
               glow=True, alpha=1.0):
    # Trail
    if trail is not None and len(trail) > 1:
        for i in range(1, len(trail)):
            fa = (i / len(trail)) * 0.55 * alpha
            ax.plot([trail[i-1][0], trail[i][0]],
                    [trail[i-1][1], trail[i][1]],
                    color=trail_color, lw=2.0, alpha=fa, solid_capstyle="round", zorder=5)

    # Glow rings
    if glow:
        for r, a in [(0.7, 0.08), (0.55, 0.15)]:
            ax.add_patch(Circle(pos, r, fc=color, ec="none", alpha=a * alpha, zorder=6))

    # Body
    ax.add_patch(Circle(pos, 0.38, fc=color, ec="white", lw=1.2, alpha=alpha, zorder=7))

    # Heading arrow
    dx = math.cos(angle) * 0.55
    dy = math.sin(angle) * 0.55
    ax.annotate("", xy=(pos[0] + dx, pos[1] + dy), xytext=(pos[0], pos[1]),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=1.5, mutation_scale=10),
                zorder=8, alpha=alpha)


def draw_goal(ax, pulse=0.0):
    r = 0.6 + 0.1 * math.sin(pulse)
    ax.add_patch(Circle(GOAL, r + 0.3, fc=YELLOW, ec="none", alpha=0.1 + 0.05 * math.sin(pulse)))
    ax.add_patch(Circle(GOAL, r,       fc=YELLOW, ec="white", lw=1.5, alpha=0.85, zorder=6))
    ax.text(GOAL[0], GOAL[1] - 1.1, "GOAL", color=YELLOW,
            fontsize=6, ha="center", fontweight="bold", alpha=0.9)


def draw_start_marker(ax):
    ax.add_patch(Circle(START, 0.5, fc=ACCENT, ec="none", alpha=0.2))
    ax.text(START[0], START[1] - 1.0, "START", color=ACCENT,
            fontsize=6, ha="center", fontweight="bold", alpha=0.8)


def render_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    # Resize to exact video dims
    if img.shape[0] != VH or img.shape[1] != VW:
        pil = Image.fromarray(img)
        pil = pil.resize((VW, VH), Image.LANCZOS)
        img = np.array(pil)
    return img


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def ease_out(t):
    return 1 - (1 - clamp(t)) ** 3

def ease_inout(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)

# ── Reward simulation ──────────────────────────────────────────────────────
def episode_reward_curve(quality, n=80):
    """Simulate per-step rewards for an episode."""
    rewards = []
    for i in range(n):
        t = i / n
        if quality == "untrained":
            r = rng.normal(-0.4, 0.3)
        elif quality == "partial":
            r = rng.normal(-0.1 + 0.4 * t, 0.2)
        else:  # trained / cultural
            r = rng.normal(0.2 + 0.6 * t, 0.1)
        rewards.append(np.clip(r, -1.5, 2.0))
    if quality in ("trained", "cultural"):
        rewards[-5:] = [1.8, 1.9, 1.95, 2.0, 2.0]  # goal reached bonus
    return np.array(rewards)

# ── Overall training reward (for the running curve panel) ─────────────────
N_EP = 300
ep_rewards = []
for ep in range(N_EP):
    q = ep / N_EP
    base = -0.8 + 2.2 * ease_out(q)
    ep_rewards.append(base + rng.normal(0, 0.15))
ep_rewards = np.array(ep_rewards)

def smooth(arr, w=12):
    if len(arr) < w:
        return arr
    k = np.ones(w) / w
    s = np.convolve(arr, k, mode="valid")
    return np.concatenate([np.full(w - 1, s[0]), s])

# ════════════════════════════════════════════════════════════════════════════
# FRAME BUILDERS
# ════════════════════════════════════════════════════════════════════════════

# ── Fade overlay ──────────────────────────────────────────────────────────
def add_fade(fig, alpha):
    if alpha <= 0:
        return
    ax = fig.add_axes([0, 0, 1, 1], zorder=999)
    ax.patch.set_facecolor(BG)
    ax.patch.set_alpha(alpha)
    ax.axis("off")

# ── SCENE 1: Title ────────────────────────────────────────────────────────
def build_title(t):
    fig = plt.figure(figsize=(VW/DPI, VH/DPI), facecolor=BG, dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")

    # Grid background
    for xi in np.linspace(0.05, 0.95, 22):
        ax.plot([xi, xi], [0, 1], color=ACCENT, alpha=0.025, lw=0.6,
                transform=ax.transAxes)
    for yi in np.linspace(0.05, 0.95, 13):
        ax.plot([0, 1], [yi, yi], color=ACCENT, alpha=0.025, lw=0.6,
                transform=ax.transAxes)

    ta = ease_out(t / 0.5)
    sa = ease_out((t - 0.3) / 0.5)
    da = ease_out((t - 0.55) / 0.4)

    # Accent line
    line_w = ease_out(t / 0.6)
    ax.plot([0.5 - line_w * 0.3, 0.5 + line_w * 0.3], [0.41, 0.41],
            transform=ax.transAxes, color=ACCENT, lw=2.5, alpha=0.8)

    ax.text(0.5, 0.60,
            "Reinforcement Learning\nHospital Robot Navigation",
            transform=ax.transAxes, color=WHITE,
            fontsize=40, fontweight="bold", ha="center", va="center",
            alpha=ta, linespacing=1.3,
            path_effects=[pe.withStroke(linewidth=6, foreground=BG)])

    ax.text(0.5, 0.36,
            "Learning to navigate a crowded hospital — from scratch",
            transform=ax.transAxes, color=ACCENT,
            fontsize=16, ha="center", va="center", alpha=sa)

    ax.text(0.5, 0.26,
            "LMIC Hospital  ·  45 pedestrians  ·  Cultural constraints",
            transform=ax.transAxes, color=GRAY,
            fontsize=12, ha="center", va="center", alpha=da)

    add_fade(fig, max(0, 1 - t / 0.35))   # fade in
    return render_fig(fig)


# ── SCENE 2: Map introduction ─────────────────────────────────────────────
def build_map_intro(t):
    fig = plt.figure(figsize=(VW/DPI, VH/DPI), facecolor=BG, dpi=DPI)

    # Map — full screen
    ax_map = fig.add_axes([0.05, 0.10, 0.58, 0.82])
    draw_map(ax_map, show_zones=True, zone_pulse=math.sin(t * 4))
    draw_pedestrians(ax_map, jitter_peds(PEDS_STATIC, t), alpha=0.7 * ease_out(t / 0.4))
    draw_goal(ax_map, t * 8)
    draw_start_marker(ax_map)

    # Info panel
    ax_info = fig.add_axes([0.66, 0.10, 0.32, 0.82])
    ax_info.set_facecolor(PANEL)
    ax_info.axis("off")
    for sp in ax_info.spines.values():
        sp.set_edgecolor(GRAY)

    items = [
        (0.82, ACCENT,  "THE ENVIRONMENT",    22, "bold"),
        (0.72, WHITE,   "40 m × 30 m hospital ward", 13, "normal"),
        (0.62, WHITE,   "Dhaka Medical College", 13, "normal"),
        (0.52, GRAY,    "─────────────────────", 9, "normal"),
        (0.44, ORANGE,  "45 pedestrians",       13, "normal"),
        (0.36, "#5588FF","Male Ward (blue)",     12, "normal"),
        (0.29, PURPLE,  "Female Ward (purple)", 12, "normal"),
        (0.22, GREEN,   "Prayer Room (green)",  12, "normal"),
        (0.14, YELLOW,  "Goal destination",     12, "normal"),
        (0.06, ACCENT,  "Robot (starts here)",  12, "normal"),
    ]
    for i, (y, col, txt, sz, wt) in enumerate(items):
        alpha = ease_out(clamp((t - i * 0.06) / 0.3))
        ax_info.text(0.08, y, txt, transform=ax_info.transAxes,
                     color=col, fontsize=sz, fontweight=wt, alpha=alpha,
                     va="center")

    ax_map.set_title("Hospital Floor Plan", color=WHITE, fontsize=12,
                     pad=6, fontweight="bold")

    add_fade(fig, max(0, 1 - t / 0.3))
    return render_fig(fig)


# ── SCENE 3: Episode playback ─────────────────────────────────────────────
def build_episode(t, traj, ep_num, ep_label, color, quality, reward_curve,
                  all_ep_up_to, fade_in=True, fade_out=True):
    """
    t: [0,1] within this scene
    traj: full trajectory array
    all_ep_up_to: index of last episode shown in training curve
    """
    n = len(traj)
    pos_idx  = min(int(t * n), n - 1)
    pos      = traj[pos_idx]
    trail_start = max(0, pos_idx - 35)
    trail    = traj[trail_start:pos_idx + 1]

    # Heading angle
    if pos_idx > 0:
        d = traj[pos_idx] - traj[max(0, pos_idx - 1)]
        angle = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-4 else math.pi / 2
    else:
        angle = math.pi / 2

    # Reward for this step
    rew_idx  = min(int(t * len(reward_curve)), len(reward_curve) - 1)
    cur_rew  = reward_curve[rew_idx]

    fig = plt.figure(figsize=(VW/DPI, VH/DPI), facecolor=BG, dpi=DPI)

    # ── Map (left 66%) ────────────────────────────────────────────────────
    ax_map = fig.add_axes([0.02, 0.08, 0.64, 0.84])
    zone_pulse = math.sin(t * 6) if quality in ("trained", "cultural") else 0.0
    draw_map(ax_map, show_zones=(quality in ("trained", "cultural")),
             zone_pulse=zone_pulse)
    draw_pedestrians(ax_map, jitter_peds(PEDS_STATIC, t), alpha=0.65)
    draw_goal(ax_map, t * 8)
    draw_robot(ax_map, pos, angle, trail=trail, color=color, trail_color=color)

    # Collision flash
    if quality == "untrained":
        near_wall = (pos[0] < 1.5 or pos[0] > 38.5 or
                     pos[1] < 1.5 or pos[1] > 28.5)
        if near_wall:
            ax_map.add_patch(Rectangle((-0.5, -0.5), 41.5, 31.5,
                                        fc=RED, ec="none", alpha=0.08))

    # Distance to goal arc
    dist = np.linalg.norm(pos - GOAL)
    ax_map.annotate("", xy=GOAL, xytext=pos,
                    arrowprops=dict(arrowstyle="-", color=YELLOW,
                                    lw=1.0, ls="--", alpha=0.25))

    ax_map.set_title(ep_label, color=color, fontsize=14,
                     pad=6, fontweight="bold",
                     path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

    # ── Right panel ───────────────────────────────────────────────────────
    ax_r = fig.add_axes([0.68, 0.08, 0.30, 0.84])
    ax_r.set_facecolor(PANEL)
    ax_r.axis("off")

    # Episode badge
    badge_alpha = ease_out(clamp(t / 0.25))
    ax_r.add_patch(FancyBboxPatch((0.05, 0.88), 0.90, 0.10,
                                   boxstyle="round,pad=0.02",
                                   fc=BG, ec=color, lw=2.0, alpha=badge_alpha,
                                   transform=ax_r.transAxes))
    ax_r.text(0.50, 0.935, f"Episode  {ep_num}", transform=ax_r.transAxes,
              color=color, fontsize=14, fontweight="bold", ha="center", alpha=badge_alpha)

    # Live reward bar
    bar_alpha = ease_out(clamp((t - 0.1) / 0.3))
    r_norm = (cur_rew + 1.5) / 3.5   # map [-1.5, 2.0] → [0,1]
    r_col  = GREEN if cur_rew > 0 else RED
    ax_r.add_patch(Rectangle((0.08, 0.76), 0.84, 0.06,
                              fc="#0A1020", ec=GRAY, lw=1, alpha=bar_alpha,
                              transform=ax_r.transAxes))
    ax_r.add_patch(Rectangle((0.08, 0.76), 0.84 * clamp(r_norm), 0.06,
                              fc=r_col, ec="none", alpha=bar_alpha * 0.9,
                              transform=ax_r.transAxes))
    ax_r.text(0.50, 0.73, f"Step Reward: {cur_rew:+.2f}",
              transform=ax_r.transAxes, color=r_col,
              fontsize=10, ha="center", alpha=bar_alpha)
    ax_r.text(0.50, 0.80, "LIVE REWARD", transform=ax_r.transAxes,
              color=GRAY, fontsize=8, ha="center", alpha=bar_alpha)

    # Distance to goal
    dist_norm = clamp(1 - dist / 25)
    ax_r.add_patch(Rectangle((0.08, 0.62), 0.84, 0.06,
                              fc="#0A1020", ec=GRAY, lw=1, alpha=bar_alpha,
                              transform=ax_r.transAxes))
    ax_r.add_patch(Rectangle((0.08, 0.62), 0.84 * dist_norm, 0.06,
                              fc=YELLOW, ec="none", alpha=bar_alpha * 0.8,
                              transform=ax_r.transAxes))
    ax_r.text(0.50, 0.59, f"Distance to Goal: {dist:.1f} m",
              transform=ax_r.transAxes, color=YELLOW,
              fontsize=10, ha="center", alpha=bar_alpha)
    ax_r.text(0.50, 0.66, "GOAL PROXIMITY", transform=ax_r.transAxes,
              color=GRAY, fontsize=8, ha="center", alpha=bar_alpha)

    # Training curve (small)
    ax_tc = fig.add_axes([0.695, 0.16, 0.27, 0.36])
    ax_tc.set_facecolor("#070F1E")
    for sp in ax_tc.spines.values():
        sp.set_edgecolor(GRAY)
        sp.set_linewidth(0.5)
    if all_ep_up_to > 1:
        xs = np.arange(all_ep_up_to)
        sm = smooth(ep_rewards[:all_ep_up_to])
        ax_tc.plot(xs, sm, color=color, lw=1.8, alpha=0.9)
        ax_tc.fill_between(xs, sm, -1.2, alpha=0.15, color=color)
        ax_tc.axhline(0, color=GRAY, lw=0.7, ls="--", alpha=0.5)
        ax_tc.scatter([all_ep_up_to - 1], [sm[-1]], color=color, s=40, zorder=5)
    ax_tc.set_xlim(0, N_EP)
    ax_tc.set_ylim(-1.2, 1.8)
    ax_tc.tick_params(colors=GRAY, labelsize=6)
    ax_tc.set_xlabel("Episode", color=GRAY, fontsize=7)
    ax_tc.set_ylabel("Reward", color=GRAY, fontsize=7)
    ax_tc.set_title("Training Progress", color=WHITE, fontsize=8, pad=3, fontweight="bold")

    # Status text at bottom
    status_texts = {
        "untrained": ["Random actions...", "No sense of direction", "Collisions frequent"],
        "partial":   ["Learning... 33%", "Goal direction improving", "Still erratic"],
        "trained":   ["Trained policy", "Navigating efficiently", "Reaching goal!"],
        "cultural":  ["Trained + Cultural", "Respecting prayer zones", "Family awareness"],
    }
    lines = status_texts.get(quality, [])
    status_colors = {
        "untrained": RED,
        "partial":   ORANGE,
        "trained":   GREEN,
        "cultural":  PURPLE,
    }
    sc = status_colors.get(quality, WHITE)
    for i, line in enumerate(lines):
        sa2 = ease_out(clamp((t - 0.2 - i * 0.08) / 0.25))
        ax_r.text(0.50, 0.10 + i * 0.055, line,
                  transform=ax_r.transAxes, color=sc if i == 0 else WHITE,
                  fontsize=10 if i == 0 else 9,
                  fontweight="bold" if i == 0 else "normal",
                  ha="center", alpha=sa2)

    # Goal reached flash
    if quality in ("trained", "cultural") and dist < 1.5:
        flash_alpha = 0.15 * max(0, math.sin(t * 20))
        ax_map.add_patch(Rectangle((-0.5, -0.5), 41.5, 31.5,
                                    fc=GREEN, ec="none", alpha=flash_alpha))
        if dist < 0.9:
            ax_r.text(0.5, 0.50, "GOAL\nREACHED", transform=ax_r.transAxes,
                      color=GREEN, fontsize=20, fontweight="bold",
                      ha="center", va="center",
                      path_effects=[pe.withStroke(linewidth=4, foreground=BG)])

    if fade_in:
        add_fade(fig, max(0, 1 - t / 0.2))
    if fade_out:
        add_fade(fig, max(0, (t - 0.85) / 0.15))
    return render_fig(fig)


# ── SCENE 4: Split comparison ─────────────────────────────────────────────
def build_comparison(t, traj_bad, traj_good):
    fig = plt.figure(figsize=(VW/DPI, VH/DPI), facecolor=BG, dpi=DPI)

    n_bad  = len(traj_bad)
    n_good = len(traj_good)
    bi = min(int(t * n_bad),  n_bad  - 1)
    gi = min(int(t * n_good), n_good - 1)

    def get_angle(traj, idx):
        if idx > 0:
            d = traj[idx] - traj[max(0, idx - 1)]
            return math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-4 else 0
        return 0

    # Left: untrained
    ax_l = fig.add_axes([0.01, 0.10, 0.47, 0.78])
    draw_map(ax_l)
    draw_pedestrians(ax_l, jitter_peds(PEDS_STATIC, t), alpha=0.6)
    draw_goal(ax_l, t * 6)
    draw_robot(ax_l, traj_bad[bi],  get_angle(traj_bad, bi),
               trail=traj_bad[max(0, bi-30):bi+1], color=RED)
    ax_l.set_title("UNTRAINED  —  Episode 1", color=RED, fontsize=13,
                   pad=5, fontweight="bold",
                   path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

    # Right: trained
    ax_r = fig.add_axes([0.52, 0.10, 0.47, 0.78])
    draw_map(ax_r, show_zones=True, zone_pulse=math.sin(t * 5))
    draw_pedestrians(ax_r, jitter_peds(PEDS_STATIC, t), alpha=0.6)
    draw_goal(ax_r, t * 6)
    draw_robot(ax_r, traj_good[gi], get_angle(traj_good, gi),
               trail=traj_good[max(0, gi-30):gi+1], color=GREEN)
    ax_r.set_title("TRAINED  —  Episode 300", color=GREEN, fontsize=13,
                   pad=5, fontweight="bold",
                   path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

    # Divider
    ax_div = fig.add_axes([0.485, 0.08, 0.03, 0.84])
    ax_div.set_facecolor(BG)
    ax_div.axis("off")
    ax_div.text(0.5, 0.5, "VS", transform=ax_div.transAxes,
                color=WHITE, fontsize=18, fontweight="bold",
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=4, foreground=BG)])

    # Bottom bar
    ax_bot = fig.add_axes([0.01, 0.01, 0.98, 0.08])
    ax_bot.set_facecolor(PANEL)
    ax_bot.axis("off")
    stats_l = [("Success Rate", "12%", RED),  ("Avg Reward", "-0.74", RED)]
    stats_r = [("Success Rate", "89%", GREEN), ("Avg Reward", "+1.42", GREEN)]
    for i, (label, val, col) in enumerate(stats_l):
        ax_bot.text(0.07 + i * 0.18, 0.65, label, transform=ax_bot.transAxes,
                    color=GRAY, fontsize=9, ha="center")
        ax_bot.text(0.07 + i * 0.18, 0.20, val, transform=ax_bot.transAxes,
                    color=col, fontsize=13, fontweight="bold", ha="center")
    for i, (label, val, col) in enumerate(stats_r):
        ax_bot.text(0.57 + i * 0.18, 0.65, label, transform=ax_bot.transAxes,
                    color=GRAY, fontsize=9, ha="center")
        ax_bot.text(0.57 + i * 0.18, 0.20, val, transform=ax_bot.transAxes,
                    color=col, fontsize=13, fontweight="bold", ha="center")
    ax_bot.text(0.5, 0.5, "300 000 training steps", transform=ax_bot.transAxes,
                color=ACCENT, fontsize=10, ha="center", va="center")

    add_fade(fig, max(0, 1 - t / 0.2))
    add_fade(fig, max(0, (t - 0.88) / 0.12))
    return render_fig(fig)


# ── SCENE 5: End card ─────────────────────────────────────────────────────
def build_endcard(t):
    fig = plt.figure(figsize=(VW/DPI, VH/DPI), facecolor=BG, dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")

    for xi in np.linspace(0.05, 0.95, 22):
        ax.plot([xi, xi], [0, 1], color=ACCENT, alpha=0.025, lw=0.6,
                transform=ax.transAxes)

    ta = ease_out(t / 0.4)
    sa = ease_out((t - 0.25) / 0.4)
    da = ease_out((t - 0.50) / 0.4)

    ax.text(0.5, 0.72, "SAC + Cultural-Adaptive Rewards",
            transform=ax.transAxes, color=ACCENT, fontsize=26, fontweight="bold",
            ha="center", alpha=ta,
            path_effects=[pe.withStroke(linewidth=5, foreground=BG)])
    ax.text(0.5, 0.60, "Robot learns to navigate a crowded LMIC hospital\nwhile respecting cultural constraints",
            transform=ax.transAxes, color=WHITE, fontsize=15, ha="center",
            va="center", linespacing=1.5, alpha=sa)

    # Stats row
    stats = [
        ("89.1%", "Success Rate", GREEN),
        ("3.8%",  "Cultural Violations", RED),
        ("300k",  "Training Steps", ACCENT),
        ("45",    "Pedestrians", ORANGE),
    ]
    for i, (val, lbl, col) in enumerate(stats):
        xa = ease_out(clamp((t - 0.45 - i * 0.07) / 0.3))
        x = 0.10 + i * 0.22
        ax.add_patch(FancyBboxPatch((x, 0.32), 0.18, 0.16,
                                    boxstyle="round,pad=0.02",
                                    fc=PANEL, ec=col, lw=1.8, alpha=xa,
                                    transform=ax.transAxes))
        ax.text(x + 0.09, 0.43, val, transform=ax.transAxes,
                color=col, fontsize=18, fontweight="bold",
                ha="center", alpha=xa)
        ax.text(x + 0.09, 0.35, lbl, transform=ax.transAxes,
                color=WHITE, fontsize=8, ha="center", alpha=xa)

    ax.text(0.5, 0.22,
            "IEEE RA-L  ·  LMIC Hospital Navigation Research",
            transform=ax.transAxes, color=GRAY, fontsize=11,
            ha="center", alpha=da)

    add_fade(fig, max(0, 1 - t / 0.35))
    add_fade(fig, max(0, (t - 0.85) / 0.15))
    return render_fig(fig)


# ════════════════════════════════════════════════════════════════════════════
# VIDEO ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════

def secs(s):
    return int(s * FPS)


def main():
    out_path = "/home/roky/RL_med_robot/lmic-hospitalnav/robot_rl_demo.mp4"
    writer = imageio.get_writer(out_path, fps=FPS, quality=9,
                                codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1,
                                output_params=["-b:v", "5M"])

    # Pre-compute reward curves
    rew_untrained = episode_reward_curve("untrained", 80)
    rew_partial   = episode_reward_curve("partial",   80)
    rew_trained   = episode_reward_curve("trained",   80)
    rew_cultural  = episode_reward_curve("cultural",  80)

    # ── Scene schedule ────────────────────────────────────────────────────
    # (name, n_frames, build_fn)
    schedule = [
        ("title",      secs(3.5),  None),
        ("map_intro",  secs(5.0),  None),
        ("ep1",        secs(11.0), None),   # untrained
        ("ep2",        secs(9.0),  None),   # partial
        ("ep3",        secs(13.0), None),   # trained, goal reached
        ("compare",    secs(9.0),  None),   # split view
        ("endcard",    secs(5.0),  None),
    ]
    total = sum(n for _, n, _ in schedule)
    print(f"Building 1-minute RL demo video...")
    print(f"  Target: {total/FPS:.1f}s  |  {total} frames  |  {FPS} fps  |  1920x1080")

    frame_n = 0

    for scene_name, n_frames, _ in schedule:
        print(f"  Rendering '{scene_name}' ({n_frames} frames, {n_frames/FPS:.1f}s)...",
              flush=True)

        for f in range(n_frames):
            t = f / n_frames   # [0,1] within scene

            if scene_name == "title":
                img = build_title(t)

            elif scene_name == "map_intro":
                img = build_map_intro(t)

            elif scene_name == "ep1":
                img = build_episode(
                    t, TRAJS["untrained"], ep_num=1,
                    ep_label="Episode 1  —  Untrained (Random Policy)",
                    color=RED, quality="untrained",
                    reward_curve=rew_untrained,
                    all_ep_up_to=max(1, int(t * 5)),
                    fade_in=True, fade_out=True)

            elif scene_name == "ep2":
                img = build_episode(
                    t, TRAJS["partial"], ep_num=100,
                    ep_label="Episode 100  —  Still Learning...",
                    color=ORANGE, quality="partial",
                    reward_curve=rew_partial,
                    all_ep_up_to=max(5, int(t * 120) + 5),
                    fade_in=True, fade_out=True)

            elif scene_name == "ep3":
                img = build_episode(
                    t, TRAJS["trained"], ep_num=300,
                    ep_label="Episode 300  —  Trained Policy",
                    color=GREEN, quality="trained",
                    reward_curve=rew_trained,
                    all_ep_up_to=min(N_EP, int(t * 60) + 240),
                    fade_in=True, fade_out=True)

            elif scene_name == "compare":
                img = build_comparison(t, TRAJS["untrained"], TRAJS["trained"])

            elif scene_name == "endcard":
                img = build_endcard(t)

            else:
                raise ValueError(scene_name)

            writer.append_data(img)
            frame_n += 1

        print(f"    Done ({frame_n}/{total})", flush=True)

    writer.close()
    print(f"\n✅  Saved: {out_path}")
    import os
    sz = os.path.getsize(out_path) / 1e6
    print(f"   Size: {sz:.1f} MB  |  {total/FPS:.1f}s  |  {total} frames")


if __name__ == "__main__":
    main()
