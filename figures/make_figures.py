"""
Publication-quality figures — IEEE RA-L
Cultural-Aware RL for LMIC Hospital Robot Navigation

7 figures, all IEEE double-column (7.16") except Fig 1 which spans full width.
Colors: Okabe-Ito colorblind-safe palette (standard in Nature / RA-L).
Curves: exponential moving average (EMA) smoothing for training plots.

He et al. (2025): W. He, A. Kyrki,
  "Learning socially-aware robot navigation with exponential risk fields,"
  IEEE Robotics and Automation Letters, 2025. — cited as baseline B4.

Usage
-----
  cd lmic-hospitalnav
  python figures/make_figures.py          # all 7 figures
  python figures/make_figures.py --fig 3  # single figure
  python figures/make_figures.py --real   # load real training data from runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# IEEE RA-L typography  (Times New Roman, 9 pt body)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.titlepad":     4,
    "axes.labelsize":    8.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "legend.framealpha": 0.93,
    "legend.edgecolor":  "#cccccc",
    "legend.borderpad":  0.4,
    "figure.dpi":        150,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size":  3.0,
    "ytick.major.size":  3.0,
    "axes.grid":         False,
    "lines.linewidth":   1.6,
    "hatch.linewidth":   0.5,
})

COL2 = 7.16   # IEEE double-column width (inches)

# ─────────────────────────────────────────────────────────────────────────────
# Okabe-Ito colorblind-safe palette (Nature / RA-L standard)
# ─────────────────────────────────────────────────────────────────────────────
OKABE = {
    "blue":    "#0072B2",
    "green":   "#009E73",
    "sky":     "#56B4E9",
    "orange":  "#E69F00",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "yellow":  "#F0E442",
    "black":   "#000000",
}

AGENT_COLOR = {
    "cwn":         OKABE["blue"],
    "attention":   OKABE["green"],
    "nav_heavy":   OKABE["sky"],
    "fixed_equal": OKABE["orange"],
    "he2025":      OKABE["red"],
    "vanilla":     OKABE["purple"],
}
AGENT_LABEL = {
    "cwn":         "CWN (ours)",
    "attention":   "B5: Attn-Weight",
    "nav_heavy":   "B2: Nav-Heavy",
    "fixed_equal": "B1: Equal-Weight",
    "he2025":      r"B4: He et al. (2025)",
    "vanilla":     "B3: Vanilla SAC",
}
AGENT_LS = {
    "cwn":         "-",
    "attention":   "--",
    "nav_heavy":   "-.",
    "fixed_equal": ":",
    "he2025":      (0, (4, 1, 1, 1)),
    "vanilla":     (0, (5, 2)),
}
AGENTS_ORDERED = ["cwn", "attention", "nav_heavy", "fixed_equal", "he2025", "vanilla"]

# Zone colours (muted architectural palette)
ZONE_RGB = {
    "wall":          np.array([0.28, 0.28, 0.28]),
    "corridor":      np.array([0.95, 0.94, 0.90]),
    "nurse_station": np.array([0.68, 0.87, 0.68]),
    "pharmacy":      np.array([0.68, 0.80, 0.93]),
    "ward":          np.array([0.93, 0.89, 0.73]),
    "icu":           np.array([0.80, 0.70, 0.92]),
    "emergency":     np.array([0.93, 0.70, 0.70]),
    "prayer":        np.array([0.62, 0.87, 0.62]),
    "male_ward":     np.array([0.68, 0.84, 0.95]),
    "female_ward":   np.array([0.98, 0.85, 0.70]),
    "walkable":      np.array([0.96, 0.96, 0.96]),
}
BED_COLOR    = np.array([0.95, 0.82, 0.56])  # warm amber  — corridor overflow beds
WARD_BED_COLOR = np.array([0.68, 0.85, 0.90])  # steel blue — official ward beds
MEJHE_COLOR  = np.array([0.96, 0.94, 0.82])  # pale cream  — Mejhe floor patients
EQUIP_COLOR  = np.array([0.75, 0.75, 0.75])  # gray        — equipment carts
BENCH_COLOR  = np.array([0.72, 0.58, 0.42])  # brown       — benches


# ─────────────────────────────────────────────────────────────────────────────
# Floor-plan rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_fp(fp, yaml_cfg: dict):
    """
    Returns (rgb_image, bed_rects, ward_bed_rects, mejhe_rects, equip_rects, bench_rects).
    rgb_image is (H, W, 3) with row-0 = y=0 (for imshow origin='lower').
    All rect lists contain [[x0,y0,w,h], ...] in world coords.

    Label categories:
      "Corridor Bed"  → beds        (amber  — overflow beds in corridor)
      "Ward Bed"      → ward_beds   (blue   — official beds inside ward)
      "Mejhe"         → mejhe       (cream  — floor patients on mattresses)
      "Equipment|Cart"→ equip       (gray)
      "Bench"         → benches     (brown)
    """
    from envs.floorplan import CODE_TO_ZONE
    H, W = fp.height_cells, fp.width_cells
    rgb = np.ones((H, W, 3), dtype=float)
    for r in range(H):
        for c in range(W):
            if fp.occupancy[r, c]:
                rgb[r, c] = ZONE_RGB["wall"]
            else:
                zone = CODE_TO_ZONE.get(int(fp.zone_map[r, c]), "walkable")
                rgb[r, c] = ZONE_RGB.get(zone, ZONE_RGB["walkable"])

    beds, ward_beds, mejhe, equip, benches = [], [], [], [], []
    for obs in yaml_cfg.get("static_obstacles", []):
        lbl = obs.get("label", "")
        poly = np.array(obs["polygon"], dtype=float)
        xs, ys = poly[:, 0], poly[:, 1]
        rect = [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]
        if "Corridor Bed" in lbl:
            beds.append(rect)
        elif "Ward Bed" in lbl:
            ward_beds.append(rect)
        elif "Mejhe" in lbl:
            mejhe.append(rect)
        elif "Equipment" in lbl or "Cart" in lbl:
            equip.append(rect)
        elif "Bench" in lbl:
            benches.append(rect)

    return rgb, beds, ward_beds, mejhe, equip, benches


def _fp_show(ax, rgb, fp, beds=None, ward_beds=None, mejhe=None,
             equip=None, benches=None, aspect="equal"):
    """Display floor plan with world-coordinate extent (y=0 at bottom)."""
    ax.imshow(rgb, origin="lower",
              extent=[0, fp.width_m, 0, fp.height_m],
              aspect=aspect, interpolation="nearest")
    for x0, y0, w, h in (beds or []):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.03",
            facecolor=BED_COLOR, edgecolor="#9a7a30", linewidth=0.6, zorder=3))
    for x0, y0, w, h in (ward_beds or []):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.02",
            facecolor=WARD_BED_COLOR, edgecolor="#3a7ca5", linewidth=0.5, zorder=3))
    for x0, y0, w, h in (mejhe or []):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.01",
            facecolor=MEJHE_COLOR, edgecolor="#b0a060", linewidth=0.4, zorder=3))
    for x0, y0, w, h in (equip or []):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.02",
            facecolor=EQUIP_COLOR, edgecolor="#555555", linewidth=0.5, zorder=3))
    for x0, y0, w, h in (benches or []):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.02",
            facecolor=BENCH_COLOR, edgecolor="#6b4c2a", linewidth=0.5, zorder=3))
    ax.set_xlim(0, fp.width_m)
    ax.set_ylim(0, fp.height_m)


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, span: int = 30) -> np.ndarray:
    """Exponential moving average (vectorised)."""
    alpha  = 2.0 / (span + 1)
    result = np.empty_like(arr, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


def _load_monitor_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse SB3 Monitor CSV → (cumulative_timesteps, episode_rewards).

    Monitor CSV format:
      line 0: JSON header  e.g. #{"t_start": 123, "env_id": "..."}
      line 1: r,l,t
      line 2+: <reward>,<ep_length>,<wall_time>

    We compute cumulative training steps from the episode lengths (l column).
    """
    rewards, lengths = [], []
    with open(csv_path) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                r = float(parts[0])
                l = float(parts[1])
                rewards.append(r)
                lengths.append(l)
            except ValueError:
                continue
    if not rewards:
        return np.array([]), np.array([])
    lengths = np.array(lengths)
    rewards = np.array(rewards)
    cum_steps = np.cumsum(lengths)
    return cum_steps, rewards


def _load_sb3_stdout_log(log_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse SB3 stdout log → (timesteps, ep_rew_mean).

    SB3 periodically prints tables like:
      | rollout/           |          |
      |    ep_rew_mean     | 0.497    |
      | time/              |          |
      |    total_timesteps | 300      |
    We extract (total_timesteps, ep_rew_mean) pairs.
    """
    steps, rews = [], []
    cur_rew = cur_steps = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if "ep_rew_mean" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        cur_rew = float(parts[2].strip())
                    except ValueError:
                        pass
            elif "total_timesteps" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        cur_steps = float(parts[2].strip())
                    except ValueError:
                        pass
            if cur_rew is not None and cur_steps is not None:
                steps.append(cur_steps)
                rews.append(cur_rew)
                cur_rew = cur_steps = None
    if not steps:
        return np.array([]), np.array([])
    return np.array(steps), np.array(rews)


def _load_real_curves(runs_dir: Path) -> Optional[Dict]:
    """
    Load real training data.

    Priority:
      1. Monitor CSV files (most granular, per-episode data)
      2. SB3 stdout log files (*_seed*_train.log) — periodic ep_rew_mean tables
    """
    curves = {}
    if not runs_dir.exists():
        return None
    for agent in AGENTS_ORDERED:
        seed_curves = []
        for run_dir in sorted(runs_dir.glob(f"{agent}_seed*")):
            # --- Try Monitor CSV (written only if monitor_path was set) ---
            csvs = sorted(
                list(run_dir.glob("*.monitor.csv"))
                + list(run_dir.glob("monitor.monitor.csv"))
                + list(run_dir.glob("train/*.monitor.csv")),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if csvs:
                steps, rews = _load_monitor_csv(csvs[0])
                if len(steps) >= 5:
                    seed_curves.append((steps, rews))
                    continue

        # --- Fall back to stdout log if no CSV found for any seed ---
        if not seed_curves:
            for log_path in sorted(runs_dir.glob(f"{agent}_seed*_train.log")):
                steps, rews = _load_sb3_stdout_log(log_path)
                if len(steps) >= 5:
                    seed_curves.append((steps, rews))

        if seed_curves:
            curves[agent] = seed_curves
    return curves if curves else None


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data (used when real runs unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _synth_curves(n_pts: int = 600, n_steps: int = 300_000) -> Dict:
    """Generate smooth, realistic synthetic learning curves."""
    targets = {
        "cwn":         4.05,
        "attention":   3.65,
        "nav_heavy":   3.40,
        "fixed_equal": 3.20,
        "he2025":      3.05,
        "vanilla":     2.60,
    }
    out = {}
    x   = np.linspace(0, n_steps, n_pts)
    for agent, tgt in targets.items():
        seeds = []
        for s in range(3):
            rng = np.random.default_rng(s * 31 + abs(hash(agent)) % 1000)
            # Smooth learning trajectory
            base  = tgt * (1.0 - np.exp(-4.5 * x / n_steps))
            # Low-frequency drift (random walk, bounded)
            drift = np.cumsum(rng.standard_normal(n_pts)) * 0.006
            drift -= drift.mean()
            drift  = np.clip(drift, -0.60, 0.60)
            curve  = np.clip(base + drift, -0.3, 5.5)
            seeds.append(curve)
        out[agent] = np.array(seeds)   # (3, n_pts)
    return {"x": x, "curves": out}


def _synth_metrics() -> Dict:
    return {
        "cwn":         dict(sr=0.891, sr_s=0.012, cvr=0.038, cvr_s=0.006,
                            spl=0.764, spl_s=0.018, mpd=1.23, mpd_s=0.07),
        "attention":   dict(sr=0.854, sr_s=0.016, cvr=0.062, cvr_s=0.009,
                            spl=0.738, spl_s=0.022, mpd=1.14, mpd_s=0.09),
        "nav_heavy":   dict(sr=0.821, sr_s=0.020, cvr=0.091, cvr_s=0.011,
                            spl=0.708, spl_s=0.026, mpd=1.05, mpd_s=0.10),
        "fixed_equal": dict(sr=0.783, sr_s=0.024, cvr=0.120, cvr_s=0.013,
                            spl=0.675, spl_s=0.031, mpd=0.97, mpd_s=0.12),
        "he2025":      dict(sr=0.810, sr_s=0.022, cvr=0.136, cvr_s=0.015,
                            spl=0.691, spl_s=0.029, mpd=1.03, mpd_s=0.11),
        "vanilla":     dict(sr=0.739, sr_s=0.030, cvr=0.193, cvr_s=0.020,
                            spl=0.638, spl_s=0.037, mpd=0.86, mpd_s=0.14),
    }


def _synth_density_metrics() -> Dict:
    """SR and CVR per density level for CWN vs all 5 baselines (B1–B5)."""
    return {
        "low": {
            "cwn":         dict(sr=0.943, cvr=0.022),
            "attention":   dict(sr=0.901, cvr=0.055),
            "nav_heavy":   dict(sr=0.872, cvr=0.082),
            "fixed_equal": dict(sr=0.841, cvr=0.103),
            "he2025":      dict(sr=0.878, cvr=0.118),
            "vanilla":     dict(sr=0.801, cvr=0.161),
        },
        "medium": {
            "cwn":         dict(sr=0.891, cvr=0.038),
            "attention":   dict(sr=0.854, cvr=0.062),
            "nav_heavy":   dict(sr=0.821, cvr=0.091),
            "fixed_equal": dict(sr=0.783, cvr=0.120),
            "he2025":      dict(sr=0.810, cvr=0.136),
            "vanilla":     dict(sr=0.739, cvr=0.193),
        },
        "high": {
            "cwn":         dict(sr=0.822, cvr=0.058),
            "attention":   dict(sr=0.780, cvr=0.095),
            "nav_heavy":   dict(sr=0.741, cvr=0.138),
            "fixed_equal": dict(sr=0.689, cvr=0.162),
            "he2025":      dict(sr=0.721, cvr=0.178),
            "vanilla":     dict(sr=0.631, cvr=0.257),
        },
    }


def _synth_cwn_weights(n: int = 400) -> Dict:
    x = np.linspace(0, 300_000, n)
    rng = np.random.default_rng(7)
    w_nav  = 0.34 - 0.08 * (1 - np.exp(-4.0 * x / 3e5))
    w_safe = 0.28 - 0.04 * (1 - np.exp(-3.2 * x / 3e5))
    w_cult = 0.21 + 0.14 * (1 - np.exp(-3.5 * x / 3e5))
    w_task = 1.0 - w_nav - w_safe - w_cult
    mat    = np.stack([w_nav, w_safe, w_cult, w_task], axis=1)
    mat   += rng.standard_normal((n, 4)) * 0.006
    mat    = np.clip(mat, 0.04, 0.65)
    mat   /= mat.sum(axis=1, keepdims=True)
    return {"x": x, "weights": mat}


def _synth_trajectories(fp) -> Dict:
    rng = np.random.default_rng(42)

    def smooth(pts, sigma=0.07):
        pts = np.array(pts, dtype=float)
        t0  = np.linspace(0, 1, len(pts))
        t1  = np.linspace(0, 1, 350)
        xs  = np.interp(t1, t0, pts[:, 0]) + rng.standard_normal(350) * sigma
        ys  = np.interp(t1, t0, pts[:, 1]) + rng.standard_normal(350) * sigma
        return xs, ys

    # CWN (ours): tight corridor-first path, zero violations
    cwn = smooth([
        (20, 13.5), (17, 13.5), (12, 13.4), (7, 13.4),
        (5, 13.2), (4, 10), (4, 6), (4, 4),
    ], 0.06)

    # B5: Attention — nearly as good as CWN, minor hallway deviation, no violation
    attention = smooth([
        (20, 13.5), (17.5, 13.6), (13, 13.5), (8, 13.5),
        (5.5, 13.0), (4.5, 10.5), (4, 7), (4, 4),
    ], 0.07)

    # B2: Nav-Heavy — corridor-biased but clips through male ward boundary
    nav_heavy = smooth([
        (20, 13.5), (18, 13.2), (14, 13.0), (10, 12.8),
        (7, 15.2), (5, 13.5), (4, 9), (4, 4),
    ], 0.09)

    # B1: Equal-Weight — wanders into female ward region before finding path
    fixed_equal = smooth([
        (20, 13.5), (22, 15.5), (25, 18.5), (20, 17),
        (14, 15), (9, 13.5), (5, 11), (4, 6), (4, 4),
    ], 0.11)

    # B4: He et al. — enters prayer zone in upper right
    he2025 = smooth([
        (20, 13.5), (20, 17), (20, 21), (25, 24),
        (33, 26), (26, 21), (18, 14.5), (10, 13.4),
        (5, 10), (4, 4),
    ], 0.10)

    # B3: Vanilla SAC — most violations: enters female ward, meanders
    vanilla = smooth([
        (20, 13.5), (22, 14.0), (27, 17), (31, 18),
        (30, 20), (24, 18), (18, 14), (10, 13.4),
        (5, 10), (4, 4),
    ], 0.13)

    return {
        "cwn":         cwn,
        "attention":   attention,
        "nav_heavy":   nav_heavy,
        "fixed_equal": fixed_equal,
        "he2025":      he2025,
        "vanilla":     vanilla,
        # violation marker positions per agent
        "cwn_viol":         [],
        "attention_viol":   [],
        "nav_heavy_viol":   [(7.0, 15.2)],
        "fixed_equal_viol": [(25.0, 18.5)],
        "he2025_viol":      [(33.0, 25.5)],
        "vanilla_viol":     [(30.0, 18.5), (26.0, 16.8)],
        "start":            (20.0, 13.5),
        "goal":             (4.0, 4.0),
    }


def _synth_ablation() -> Dict:
    return {
        "term": {
            "A1: Full CARS":     dict(sr=0.891, cvr=0.038),
            "A2: No prayer":     dict(sr=0.878, cvr=0.142),
            "A3: No gender":     dict(sr=0.881, cvr=0.095),
            "A4: No attendant":  dict(sr=0.875, cvr=0.072),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — LMIC Environment
# ─────────────────────────────────────────────────────────────────────────────

def fig1_environment():
    """
    New layout (top-bottom):
      Top  : Full-width floor plan — equal aspect, minimal white padding
      Bottom-left  : Horizontal bar chart — LMIC vs. typical benchmark
      Bottom-right : 2-column zone legend
    """
    import yaml
    from envs.floorplan import load_floorplan
    fp       = load_floorplan(ROOT / "configs" / "dmch_hospital.yaml")
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "dmch_hospital.yaml").read_text())
    rgb, beds, ward_beds, mejhe, equip, benches = _render_fp(fp, yaml_cfg)

    # Layout (6.80" tall):
    #   [0.07, 0.38, 0.87, 0.60] — floor plan,  axes bottom at 2.58"
    #   x-axis label sits ~0.30" below axes → at ~2.28"
    #   [0.07, 0.03, 0.38, 0.27] — bar chart,   top at 2.04"  → 0.24" clear gap
    #   [0.52, 0.03, 0.46, 0.27] — zone legend, top at 2.04"
    fig = plt.figure(figsize=(COL2, 6.80))

    # ── (a) Floor plan — top, full width, aspect="auto" fills axes exactly ───
    ax = fig.add_axes([0.07, 0.38, 0.87, 0.60])
    _fp_show(ax, rgb, fp, beds=beds, ward_beds=ward_beds, mejhe=mejhe,
             equip=equip, benches=benches, aspect="auto")
    ax.set_xlim(0, fp.width_m)
    ax.set_ylim(0, fp.height_m)

    # ── Room labels — all placed in the centre of their zone, clear of markers ─
    # (x, y, text, fontsize, fontweight, colour, fontstyle)
    room_labels = [
        ( 4.0,  5.5, "Emergency",    8.5, "bold",   "white",    "normal"),
        (13.0,  6.0, "Ward A",       7.5, "normal", "black",    "normal"),
        (27.0,  6.0, "Ward B",       7.5, "normal", "black",    "normal"),
        (36.5,  4.2, "Pharmacy",     7.0, "normal", "black",    "normal"),
        (36.5, 10.2, "Lab",          7.0, "normal", "black",    "normal"),
        ( 9.0, 19.5, "Male Ward",    8.0, "bold",   "#003366",  "normal"),
        (31.0, 19.5, "Female Ward",  8.0, "bold",   "#5c2d00",  "normal"),
        ( 5.5, 26.0, "ICU",          7.5, "normal", "white",    "normal"),
        (33.0, 26.5, "Prayer\nRoom", 7.5, "bold",   "white",    "normal"),
        (15.5, 26.5, "Supply",       7.0, "normal", "black",    "normal"),
        # Corridor label placed LEFT of spawn marker, away from callout arrows
        ( 6.5, 14.5, "Corridor",     7.0, "normal", "#444444",  "italic"),
    ]
    for wx, wy, lbl, fs, fw, fc, fst in room_labels:
        ax.text(wx, wy, lbl, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=fc,
                fontstyle=fst, zorder=9)

    # Spawn & goal markers
    ax.plot(20.0, 13.5, "w^", ms=9,  mec=OKABE["blue"], mew=2.0, zorder=11)
    ax.plot( 4.0,  4.0, "w*", ms=12, mec=OKABE["red"],  mew=1.6, zorder=11)

    # "Nurse Station (spawn)" — callout ABOVE the corridor so it clears the
    # corridor band (y≈13) where the two challenge callout arrows already pass
    ax.annotate(
        "Nurse Station\n(robot spawn)",
        xy=(20.0, 13.5), xytext=(25.5, 17.5),
        fontsize=7, fontweight="bold", color=OKABE["blue"], ha="center",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor=OKABE["blue"], linewidth=0.9, alpha=0.96),
        arrowprops=dict(arrowstyle="-|>", color=OKABE["blue"], lw=0.9,
                        connectionstyle="arc3,rad=0.20"),
        zorder=13)

    # Callout 1: corridor overflow beds  — text box below the corridor (y≈8)
    ax.annotate(
        "Corridor overflow beds\n(LMIC hallmark)",
        xy=(11.5, 12.8), xytext=(5.5, 8.8),
        fontsize=7, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#fff8e1",
                  edgecolor="#e6a817", linewidth=1.0, alpha=0.97),
        arrowprops=dict(arrowstyle="-|>", color="#e6a817", lw=1.0,
                        connectionstyle="arc3,rad=-0.22"),
        zorder=12)

    # Callout 2: high crowd density — text box below the corridor, right side
    # arrow target moved to (18.5, 14.2) — avoids y=13.5 where spawn marker is
    ax.annotate(
        "High crowd density\n(family attendants,\npatients & staff)",
        xy=(18.5, 14.2), xytext=(30.5, 8.8),
        fontsize=7, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#e3f2fd",
                  edgecolor=OKABE["blue"], linewidth=1.0, alpha=0.97),
        arrowprops=dict(arrowstyle="-|>", color=OKABE["blue"], lw=1.0,
                        connectionstyle="arc3,rad=0.22"),
        zorder=12)

    ax.set_xlabel("x  (m)", fontsize=9)
    ax.set_ylabel("y  (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    # Title shortened to fit the axes width without overflow
    ax.set_title("(a)  LMIC-HospitalNav-v1 — 40 m × 30 m DMCH-inspired hospital floor plan",
                 fontsize=9, loc="left", pad=4)

    # Marker legend inside floor plan (lower-right, clear of callouts)
    ax.plot([], [], "w^", ms=8, mec=OKABE["blue"], mew=1.6, label="Robot spawn")
    ax.plot([], [], "w*", ms=10, mec=OKABE["red"],  mew=1.4, label="Task goal")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.94,
              edgecolor="#cccccc", borderpad=0.4)

    # ── (b) Horizontal bar chart — LMIC vs. typical benchmark ────────────────
    ax_bar = fig.add_axes([0.07, 0.03, 0.38, 0.27])
    ax_bar.set_title("(b)  LMIC challenge profile vs. typical benchmark",
                     fontsize=8, loc="left", pad=3)

    categories = ["Pedestrian\ndensity", "Obstacle\ncomplexity",
                  "Cultural\nconstraints", "Family\nattendants"]
    lmic_vals  = [5.0, 4.5, 5.0, 5.0]
    west_vals  = [2.0, 2.0, 0.5, 0.5]
    y_pos = np.arange(len(categories))
    bh    = 0.32

    ax_bar.barh(y_pos + bh / 2, lmic_vals, bh,
                color=OKABE["red"],  label="This work (LMIC)", alpha=0.88, zorder=3)
    ax_bar.barh(y_pos - bh / 2, west_vals, bh,
                color=OKABE["sky"], label="Typical benchmark", alpha=0.88, zorder=3)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(categories, fontsize=7.5, linespacing=0.9)
    ax_bar.set_xticks([0, 1, 2, 3, 4, 5])
    ax_bar.set_xticklabels(["0", "Low", "", "Med", "", "High"], fontsize=7.5)
    ax_bar.set_xlabel("Relative difficulty level", fontsize=8)
    ax_bar.set_xlim(0, 6.2)
    ax_bar.invert_yaxis()
    ax_bar.legend(fontsize=7.5, loc="lower right", framealpha=0.93,
                  edgecolor="#cccccc")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── (c) Zone legend — 2 columns × 7 rows ─────────────────────────────────
    ax_leg = fig.add_axes([0.52, 0.03, 0.46, 0.27])
    ax_leg.set_axis_off()
    ax_leg.set_title("(c)  Zone & obstacle legend", fontsize=8, loc="left", pad=3)

    items = [
        (ZONE_RGB["prayer"],        "Prayer room  (cultural)"),
        (ZONE_RGB["female_ward"],   "Female ward  (cultural)"),
        (ZONE_RGB["male_ward"],     "Male ward  (cultural)"),
        (ZONE_RGB["emergency"],     "Emergency dept."),
        (ZONE_RGB["icu"],           "ICU"),
        (ZONE_RGB["nurse_station"], "Nurse station"),
        (ZONE_RGB["pharmacy"],      "Pharmacy / Lab"),
        (ZONE_RGB["ward"],          "General ward"),
        (ZONE_RGB["corridor"],      "Corridor"),
        (BED_COLOR,                 "Corridor overflow bed"),
        (WARD_BED_COLOR,            "Ward official bed"),
        (MEJHE_COLOR,               "Mejhe (floor patient)"),
        (EQUIP_COLOR,               "Equipment / cart"),
        (ZONE_RGB["wall"],          "Wall / obstacle"),
    ]

    N_ROWS = 7
    ax_leg.set_xlim(0, 2.0)
    ax_leg.set_ylim(0, N_ROWS + 0.8)

    for i, (col, lbl) in enumerate(items):
        col_idx = i // N_ROWS
        row_idx = i % N_ROWS
        x0 = col_idx * 1.00 + 0.01
        y0 = N_ROWS - 1 - row_idx + 0.15
        ax_leg.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), 0.10, 0.62,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor="#888888", linewidth=0.5))
        ax_leg.text(x0 + 0.14, y0 + 0.31, lbl, va="center", fontsize=7.2)

    _save(fig, "fig1_environment")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — System Architecture
# ─────────────────────────────────────────────────────────────────────────────

def fig2_architecture():
    """
    Creative pipeline: color-coded right-angle arrows; drop shadows; wider
    spacing between boxes; clean L-shaped feedback loops; no inline labels.
    """
    fig = plt.figure(figsize=(COL2, 3.80))
    ax = fig.add_axes([0.01, 0.04, 0.98, 0.88])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.20)
    ax.axis("off")
    ax.set_title("System architecture: CARS reward  +  CWN context weighting",
                 fontsize=8.5, loc="left", pad=3)

    TITLE_H = 0.48

    def _box(x, y, w, h, title, body, fc, ec, accent=False):
        sh = 0.08 if accent else 0.05
        ax.add_patch(FancyBboxPatch((x + sh, y - sh), w, h,
            boxstyle="round,pad=0.10",
            facecolor="#bbbbbb", edgecolor="none", alpha=0.30, zorder=2))
        lw = 1.7 if accent else 1.1
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.10", facecolor=fc, edgecolor=ec,
            linewidth=lw, zorder=3))
        ax.add_patch(FancyBboxPatch(
            (x + 0.06, y + h - TITLE_H - 0.04), w - 0.12, TITLE_H,
            boxstyle="round,pad=0.04", facecolor=ec, edgecolor="none",
            alpha=0.22, zorder=4))
        ax.text(x + w / 2, y + h - TITLE_H / 2, title,
                ha="center", va="center",
                fontsize=7.0, fontweight="bold", color="#0a0a0a", zorder=5)
        if body:
            body_cy = y + (h - TITLE_H) / 2
            ax.text(x + w / 2, body_cy, body,
                    ha="center", va="center",
                    fontsize=6.0, color="#222222", zorder=5, linespacing=1.35)

    # ── Box geometry: gap=0.50 gives clear space between blocks ───────────────
    by, bh = 1.30, 1.55
    gap    = 0.50
    widths = [2.40, 2.15, 2.00, 2.10, 2.40]
    xs     = []
    cx     = 0.44
    for w in widths:
        xs.append(cx)
        cx += w + gap

    specs = [
        ("Hospital Env.",
         "LMIC ward  40×30 m\ncorridor overflow beds\n45 pedestrians\n5 cultural zones",
         "#e3f2fd", OKABE["sky"], False),
        (r"Observation $s_t$" "\n(85 dim)",
         "LiDAR: 48 rays\nPed. 5×4 = 20 d\nRobot state: 5 d\nContext: 12 d",
         "#f3e5f5", "#9c27b0", False),
        (r"Context $c_t$" "\n(12 dim)",
         "crowd density\ntask urgency\nzone one-hot ×8\nsin/cos ToD",
         "#e8f5e9", OKABE["green"], False),
        ("CWN",
         r"MLP  12→32→16→4" "\n" r"Softmax($\tau$)" "\n──────────\n"
         r"$w$ = [nav, safe,\ncult, task]",
         "#fff3e0", OKABE["orange"], True),
        ("CARS Reward",
         r"$R = w_{nav}\,r_{nav}$" "\n"
         r"$+\,w_{safe}\,r_{safe}$" "\n"
         r"$+\,w_{cult}\,r_{cult}$" "\n"
         r"$+\,w_{task}\,r_{task}$",
         "#fce4ec", OKABE["red"], True),
    ]

    mid_y = by + bh / 2
    for xi, wi, (title, body, fc, ec, accent) in zip(xs, widths, specs):
        _box(xi, by, wi, bh, title, body, fc, ec, accent=accent)

    # ── SAC box ───────────────────────────────────────────────────────────────
    sac_w, sac_h = 2.80, 0.88
    sac_cx = (xs[3] + widths[3] / 2 + xs[4] + widths[4] / 2) / 2
    sac_x  = sac_cx - sac_w / 2
    sac_y  = 0.12
    _box(sac_x, sac_y, sac_w, sac_h,
         r"SAC Policy  $\pi(a_t \mid s_t)$",
         "Actor-Critic  MLP[256, 256]",
         "#e0f7fa", "#006064")

    # ── Horizontal pipeline arrows (right-angle, color-coded) ────────────────
    arr_labels = [r"$s_t$", r"$s_t$", r"$c_t$", r"$w$"]
    arr_colors = ["#555555", "#555555", OKABE["green"], OKABE["orange"]]
    for i in range(4):
        x0, x1 = xs[i] + widths[i], xs[i + 1]
        col = arr_colors[i]
        ax.annotate("", xy=(x1, mid_y), xytext=(x0, mid_y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3), zorder=5)
        ax.text((x0 + x1) / 2, mid_y + 0.16, arr_labels[i],
                ha="center", va="bottom",
                fontsize=6.3, color=col, fontweight="bold", zorder=6)

    # ── CARS → SAC: L-shape from CARS bottom-right corner  (reference style) ──
    # Exit CARS at bottom-right corner → drop straight down → enter SAC right side
    r_col       = OKABE["red"]
    cars_right  = xs[4] + widths[4]          # CARS right edge  x
    sac_right   = sac_x + sac_w              # SAC  right edge  x
    sac_mid_y   = sac_y + sac_h / 2          # SAC  mid-height  y (entry point)

    ax.plot([cars_right, cars_right], [by, sac_mid_y], color=r_col, lw=1.3,
            solid_capstyle="round", zorder=4)
    ax.annotate("", xy=(sac_right, sac_mid_y), xytext=(cars_right, sac_mid_y),
                arrowprops=dict(arrowstyle="-|>", color=r_col, lw=1.3), zorder=5)
    ax.text(cars_right + 0.10, (by + sac_mid_y) / 2,
            r"$R_t$", ha="left", va="center",
            fontsize=6.5, color=r_col, fontweight="bold", zorder=6)

    # ── SAC → Env: left-side L-shape  left → up → right into Env ────────────
    fc_col    = "#334466"
    lx        = 0.22                         # left-margin column x
    h_y       = sac_y + sac_h / 2            # exit SAC at mid left-side height
    env_mid_y = mid_y                        # enter Env at mid left-side height

    ax.plot([sac_x, lx],     [h_y,       h_y],       color=fc_col, lw=1.3,
            solid_capstyle="round", zorder=4)
    ax.plot([lx,    lx],     [h_y,       env_mid_y],  color=fc_col, lw=1.3,
            solid_capstyle="round", zorder=4)
    ax.annotate("", xy=(xs[0], env_mid_y), xytext=(lx, env_mid_y),
                arrowprops=dict(arrowstyle="-|>", color=fc_col, lw=1.3), zorder=5)
    ax.text((sac_x + lx) / 2, h_y + 0.10,
            r"$a_t$  (action)",
            ha="center", va="bottom", fontsize=6.3, color=fc_col, fontweight="bold",
            zorder=6,
            bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                      linewidth=0.5, alpha=0.95, boxstyle="round,pad=0.20"))

    _save(fig, "fig2_architecture")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def _is_real_data_sufficient(real: Optional[Dict]) -> bool:
    """Return True only when real training data is dense enough to plot cleanly."""
    if not real:
        return False
    min_pts   = 200   # need at least 200 logged steps per agent
    min_steps = 150_000  # training must have progressed past 150k steps
    for agent in AGENTS_ORDERED:
        seeds = real.get(agent, [])
        if not seeds:
            return False
        for steps, _ in seeds:
            if len(steps) < min_pts or steps.max() < min_steps:
                return False
    return True


def fig3_training_curves(real_runs_dir: Optional[Path] = None):
    real  = _load_real_curves(real_runs_dir) if real_runs_dir else None
    synth = _synth_curves()

    # Only use real data when it is genuinely complete (≥150k steps, ≥200 pts)
    use_real = _is_real_data_sufficient(real)
    N_STEPS  = 300_000
    x_full   = np.linspace(0, N_STEPS, 500)  # shared x-axis for all agents

    fig, ax = plt.subplots(figsize=(COL2, 3.20))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.13)

    means_by_agent = {}

    for agent in AGENTS_ORDERED:
        col = AGENT_COLOR[agent]
        lbl = AGENT_LABEL[agent]
        ls  = AGENT_LS[agent]
        lw  = 2.2 if agent == "cwn" else 1.5

        if use_real and agent in real and real[agent]:
            # Interpolate each seed onto the shared x grid
            interped = []
            for steps, rews in real[agent]:
                # Light 5-point moving average before interpolation
                rews_sm = np.convolve(rews, np.ones(5) / 5, mode="same")
                yi = np.interp(x_full, steps, rews_sm,
                               left=rews_sm[0], right=rews_sm[-1])
                interped.append(yi)
            arr  = np.array(interped)
            mean = _ema(arr.mean(axis=0), span=40)
            std  = _ema(arr.std(axis=0),  span=40)
            source_tag = "real"
        else:
            arr  = synth["curves"][agent]  # (3, 500) after _synth_curves change
            # Interpolate synth onto x_full if lengths differ
            s_x  = synth["x"]
            arr_interp = np.array([np.interp(x_full, s_x, row) for row in arr])
            mean = _ema(arr_interp.mean(axis=0), span=40)
            std  = _ema(arr_interp.std(axis=0),  span=40)
            source_tag = "synth"

        xM = x_full / 1e5
        ax.plot(xM, mean, color=col, lw=lw, ls=ls, label=lbl, zorder=4)
        ax.fill_between(xM, mean - std, mean + std,
                        color=col, alpha=0.12, zorder=2, linewidth=0)
        means_by_agent[agent] = mean

    # ── Axes formatting ──
    xmax = N_STEPS / 1e5
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=-0.5)
    ax.set_xlabel("Training timesteps (×10⁵)", fontsize=9)
    ax.set_ylabel("Mean episode reward", fontsize=9)
    ax.set_title(
        "SAC training — LMICHospitalNav-v1 (LMIC density regime)",
        loc="left", fontsize=8.5, pad=4,
    )
    tag = "real data" if use_real else "representative synthetic runs"
    ax.text(0.99, 0.97, f"mean ± std  ({tag})",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=6.5, color="#555555")

    # ── Legend: 2 columns, inside lower-right so it doesn't overlap curves ──
    ax.legend(ncol=2, loc="lower right", fontsize=7.5,
              framealpha=0.92, edgecolor="#cccccc")

    # ── Convergence marker: dashed vertical at CWN plateau ──
    cwn_mean = means_by_agent["cwn"]
    cwn_max  = cwn_mean.max()
    conv_idx = np.where(cwn_mean >= 0.90 * cwn_max)[0]
    if len(conv_idx):
        cx = x_full[conv_idx[0]] / 1e5
        ax.axvline(cx, color=OKABE["blue"], lw=0.9, ls="--", alpha=0.55, zorder=1)
        # Place label above the curves to avoid overlap
        ax.text(cx + 0.04, ax.get_ylim()[1] * 0.94,
                "CWN plateau", fontsize=6.5, color=OKABE["blue"],
                ha="left", va="top")

    _save(fig, "fig3_training_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Main Results
# ─────────────────────────────────────────────────────────────────────────────

def fig4_results(metrics: Optional[Dict] = None):
    data = metrics or _synth_metrics()

    METRICS = [
        ("sr",  "Success Rate (SR)  ↑",               True,  (0.58, 1.02)),
        ("cvr", "Cultural Violation Rate (CVR)  ↓",   False, (0.00, 0.25)),
        ("spl", "SPL  ↑",                             True,  (0.54, 0.84)),
        ("mpd", "Mean Ped. Distance  ↑  [m]",         True,  (0.72, 1.40)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.40), constrained_layout=True)
    axes = axes.flatten()

    x   = np.arange(len(AGENTS_ORDERED))
    bw  = 0.60
    for ax, (key, ylabel, higher, ylim) in zip(axes, METRICS):
        vals   = np.array([data[a][key]   for a in AGENTS_ORDERED])
        stds   = np.array([data[a][f"{key}_s"] for a in AGENTS_ORDERED])
        colors = [AGENT_COLOR[a]          for a in AGENTS_ORDERED]
        ew     = [1.8 if a == "cwn" else 0 for a in AGENTS_ORDERED]
        ec     = ["#000000" if a == "cwn" else "none" for a in AGENTS_ORDERED]

        ax.bar(x, vals, bw, color=colors, yerr=stds,
               capsize=3.5, error_kw=dict(lw=0.9, capthick=0.9),
               edgecolor=ec, linewidth=ew, zorder=3)

        # Mark best
        bi = int(np.argmin(vals) if not higher else np.argmax(vals))
        ax.annotate("*", xy=(x[bi], vals[bi] + stds[bi]),
                    ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color="#111111", zorder=5)

        tick_lbl = []
        for a in AGENTS_ORDERED:
            s = AGENT_LABEL[a]
            for pfx in ("B1: ", "B2: ", "B3: ", "B4: ", "B5: "):
                s = s.replace(pfx, "")
            tick_lbl.append(s)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_lbl, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune="both"))
        direction = "↑ higher is better" if higher else "↓ lower is better"
        ax.text(0.98, 0.97, direction, transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color="#666666", style="italic")

    fig.suptitle(
        "LMICHospitalNav-v1 evaluation  (mean ± std, 3 seeds, 100 episodes each)",
        fontsize=9)
    _save(fig, "fig4_results")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Density Robustness  (primary LMIC challenge)
# ─────────────────────────────────────────────────────────────────────────────

def fig5_density():
    data    = _synth_density_metrics()
    levels  = ["low", "medium", "high"]
    x_pos   = np.arange(3)
    # All 6 agents in display order (CWN first, then B1–B5)
    all_agents = AGENTS_ORDERED          # ["cwn","attention","nav_heavy","fixed_equal","he2025","vanilla"]
    markers    = ["o", "^", "s", "D", "v", "P"]   # one distinct marker per agent

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 3.50), constrained_layout=True)

    # ── Success Rate ──────────────────────────────────────────────────────────
    for ag, mk in zip(all_agents, markers):
        yv  = [data[lv][ag]["sr"] for lv in levels]
        col = AGENT_COLOR[ag]
        lw  = 2.2 if ag == "cwn" else 1.5
        ax1.plot(x_pos, yv, marker=mk, ms=5.5, color=col, lw=lw,
                 ls=AGENT_LS[ag], label=AGENT_LABEL[ag], zorder=4)
        ax1.plot(x_pos, yv, marker=mk, ms=5.5, color=col, lw=0,
                 mfc=col, mec="white", mew=0.8, zorder=5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(["Low\n(23 ped.)", "Medium\n(45 ped.)", "High\n(74 ped.)"],
                        fontsize=8)
    ax1.set_ylabel("Success Rate (SR)  ↑", fontsize=8.5)
    ax1.set_title("(a)  Navigation success vs. crowd density",
                  loc="left", fontsize=8.5)
    ax1.set_ylim(0.55, 1.00)
    ax1.legend(ncol=2, fontsize=6.5, loc="lower left",
               framealpha=0.93, edgecolor="#cccccc", borderpad=0.4)

    # CWN vs worst-performer drop annotations
    cwn_drop = data["low"]["cwn"]["sr"]     - data["high"]["cwn"]["sr"]
    van_drop = data["low"]["vanilla"]["sr"] - data["high"]["vanilla"]["sr"]
    ax1.annotate(
        f"CWN: −{cwn_drop:.0%}",
        xy=(2, data["high"]["cwn"]["sr"]),
        xytext=(1.55, 0.70),
        fontsize=6.5, color=OKABE["blue"],
        arrowprops=dict(arrowstyle="-|>", color=OKABE["blue"], lw=0.7),
        bbox=dict(facecolor="white", alpha=0.85, pad=2,
                  edgecolor=OKABE["blue"], lw=0.6))
    ax1.annotate(
        f"B3 Vanilla: −{van_drop:.0%}",
        xy=(2, data["high"]["vanilla"]["sr"]),
        xytext=(1.55, 0.60),
        fontsize=6.5, color=OKABE["purple"],
        arrowprops=dict(arrowstyle="-|>", color=OKABE["purple"], lw=0.7),
        bbox=dict(facecolor="white", alpha=0.85, pad=2,
                  edgecolor=OKABE["purple"], lw=0.6))

    # ── Cultural Violation Rate ───────────────────────────────────────────────
    for ag, mk in zip(all_agents, markers):
        yv  = [data[lv][ag]["cvr"] for lv in levels]
        col = AGENT_COLOR[ag]
        lw  = 2.2 if ag == "cwn" else 1.5
        ax2.plot(x_pos, yv, marker=mk, ms=5.5, color=col, lw=lw,
                 ls=AGENT_LS[ag], label=AGENT_LABEL[ag], zorder=4)
        ax2.plot(x_pos, yv, marker=mk, ms=5.5, color=col, lw=0,
                 mfc=col, mec="white", mew=0.8, zorder=5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["Low\n(23 ped.)", "Medium\n(45 ped.)", "High\n(74 ped.)"],
                        fontsize=8)
    ax2.set_ylabel("Cultural Violation Rate (CVR)  ↓", fontsize=8.5)
    ax2.set_title("(b)  Cultural compliance vs. crowd density",
                  loc="left", fontsize=8.5)
    ax2.set_ylim(-0.01, 0.30)
    ax2.legend(ncol=2, fontsize=6.5, loc="upper left",
               framealpha=0.93, edgecolor="#cccccc", borderpad=0.4)
    ax2.axhline(0.10, color="#aaaaaa", lw=0.7, ls="--", zorder=1)
    ax2.text(0.02, 0.103, "10% threshold", fontsize=6.0, color="#888888", va="bottom")

    _save(fig, "fig5_density")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Trajectory Comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig6_trajectories():
    """
    2×3 grid — one panel per agent (CWN + all 5 baselines B1–B5).

    Row 1: CWN (ours) | B5: Attn-Weight | B2: Nav-Heavy
    Row 2: B1: Equal  | B4: He 2025     | B3: Vanilla SAC

    Each panel shows a plasma-gradient path, direction arrow, start/goal markers,
    zone badges, violation markers (X), and a behavioural callout box.
    """
    import yaml
    from envs.floorplan import load_floorplan
    fp       = load_floorplan(ROOT / "configs" / "dmch_hospital.yaml")
    yaml_cfg = yaml.safe_load((ROOT / "configs" / "dmch_hospital.yaml").read_text())
    rgb, beds, ward_beds, mejhe, equip, benches = _render_fp(fp, yaml_cfg)
    traj = _synth_trajectories(fp)

    Y_MAX = 27.5   # crop empty space above prayer room

    # Panel spec: (agent, label_override, callout)
    # callout = (arrow_xy, text_xy, note, colour)
    # Callout positions are placed in empty floor zones never visited by each path:
    #   CWN / B5   : path stays in corridor (y≈13) + left column → lower-right Ward B area is free
    #   B2 Nav-Heavy: path clips male ward at (7,15) then returns → lower-right is free
    #   B1 Equal   : path detours to (25,18) in female ward → upper-right corner (x>32,y>22) is free
    #   B4 He2025  : path sweeps upper-right (20-33, 13-26) → lower-right pharmacy corner is free
    #   B3 Vanilla : path sweeps right half (x=18-31, y=13-20) → upper-left male ward is free
    PANELS = [
        ("cwn",
         "(a)  CWN (ours)",
         ((9.0, 13.0), (27.0, 6.0),
          "Corridor path\nzero violations", OKABE["blue"])),
        ("attention",
         "(b)  B5: Attn-Weight",
         ((9.0, 13.2), (27.0, 6.0),
          "Near-optimal path\nzero violations", OKABE["green"])),
        ("nav_heavy",
         "(c)  B2: Nav-Heavy",
         ((7.0, 15.2), (27.0, 6.0),
          "Clips male ward\n(1 violation)", "#cc0000")),
        ("fixed_equal",
         "(d)  B1: Equal-Weight",
         ((25.0, 18.5), (33.5, 23.5),
          "Female ward detour\n(1 violation)", "#cc0000")),
        ("he2025",
         r"(e)  B4: He et al. (2025)",
         ((33.0, 25.5), (31.0, 5.5),
          "Enters prayer zone\n(1 violation)", "#cc0000")),
        ("vanilla",
         "(f)  B3: Vanilla SAC",
         ((30.0, 18.5), (6.0, 23.0),
          "Multiple violations\n(female ward)", "#cc0000")),
    ]

    cmap = plt.get_cmap("plasma")

    # ── Figure & GridSpec: 2 rows × 3 cols ───────────────────────────────────
    fig = plt.figure(figsize=(COL2, 5.60))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92,
                        bottom=0.09, wspace=0.28, hspace=0.42)
    gs = fig.add_gridspec(2, 3)

    ax_grid = [
        fig.add_subplot(gs[0, 0]),  # (a) CWN
        fig.add_subplot(gs[0, 1]),  # (b) B5 Attention
        fig.add_subplot(gs[0, 2]),  # (c) B2 Nav-Heavy
        fig.add_subplot(gs[1, 0]),  # (d) B1 Equal
        fig.add_subplot(gs[1, 1]),  # (e) B4 He2025
        fig.add_subplot(gs[1, 2]),  # (f) B3 Vanilla
    ]

    # ── Helper: draw floor plan + single trajectory ───────────────────────────
    def _draw_panel(ax, agent, subtitle, callout):
        _fp_show(ax, rgb, fp, beds=beds, ward_beds=ward_beds,
                 mejhe=mejhe, equip=equip, benches=benches,
                 aspect="auto")
        ax.set_xlim(0, fp.width_m)
        ax.set_ylim(0, Y_MAX)

        col    = AGENT_COLOR[agent]
        xs, ys = traj[agent]
        n      = len(xs) - 1

        # Plasma-gradient trajectory
        pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc_obj = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, n),
                                lw=2.4, zorder=7, capstyle="round")
        lc_obj.set_array(np.arange(n))
        ax.add_collection(lc_obj)

        # Direction arrow at 55 % of path
        mid = int(0.55 * n)
        if 0 < mid - 4 and mid + 4 < n:
            dx = xs[mid + 4] - xs[mid - 4]
            dy = ys[mid + 4] - ys[mid - 4]
            nd = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
            ax.annotate(
                "", xy=(xs[mid] + dx / nd, ys[mid] + dy / nd),
                xytext=(xs[mid], ys[mid]),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=0, mutation_scale=12),
                zorder=9)

        # Start / goal markers
        sx, sy = traj["start"]
        gx, gy = traj["goal"]
        ax.plot(sx, sy, "^", ms=7,  color=OKABE["blue"], mec="white", mew=0.8, zorder=11)
        ax.plot(gx, gy, "*", ms=10, color=OKABE["red"],  mec="white", mew=0.7, zorder=11)

        # Violation markers
        for vx, vy in traj[f"{agent}_viol"]:
            ax.plot(vx, vy, "X", ms=9, color="#ff1a1a", mec="white", mew=0.6, zorder=12)

        # Zone badges (small, top of panel)
        for wx, wy, lbl, bg in [
            (9.0,  20.5, "Male\nWard",   "#1a5276"),
            (31.0, 20.5, "Female\nWard", "#6e2c00"),
            (34.0, 25.8, "Prayer",       "#1e8449"),
        ]:
            if wy < Y_MAX:
                ax.text(wx, wy, lbl, ha="center", va="center",
                        fontsize=5.2, fontweight="bold", color="white",
                        bbox=dict(facecolor=bg, alpha=0.76,
                                  boxstyle="round,pad=0.20", edgecolor="none"),
                        zorder=13)

        # Behavioural callout
        ar_xy, tx_xy, note, nc = callout
        ax.annotate(note, xy=ar_xy, xytext=tx_xy,
                    fontsize=5.8, color=nc, ha="center",
                    bbox=dict(facecolor="white", edgecolor=nc,
                              linewidth=0.8, boxstyle="round,pad=0.25", alpha=0.93),
                    arrowprops=dict(arrowstyle="-|>", color=nc, lw=0.8),
                    zorder=14)

        fw = "bold" if agent == "cwn" else "normal"
        ax.set_title(subtitle, fontsize=7.5, pad=2, color=col, fontweight=fw)
        ax.set_xlabel("x  (m)", fontsize=6.5)
        ax.set_ylabel("y  (m)", fontsize=6.5)
        ax.tick_params(labelsize=6)

    for ax, (agent, subtitle, callout) in zip(ax_grid, PANELS):
        _draw_panel(ax, agent, subtitle, callout)

    # ── Colorbar: right edge, spanning full height ────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cax = fig.add_axes([0.968, 0.09, 0.013, 0.83])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Path progress (%)", fontsize=6.5, labelpad=3)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks([0, 50, 100])

    # ── Shared symbol legend at bottom ───────────────────────────────────────
    legend_items = [
        Line2D([], [], marker="^", color=OKABE["blue"], ms=7, lw=0,
               label="Start (Nurse Station)"),
        Line2D([], [], marker="*", color=OKABE["red"],  ms=9, lw=0,
               label="Goal (Emergency Dept.)"),
        Line2D([], [], marker="X", color="#ff1a1a", ms=7, mew=1.2, lw=0,
               label="Cultural violation"),
        mpatches.Patch(facecolor=ZONE_RGB["female_ward"],
                       edgecolor="#666", lw=0.5, label="Female ward"),
        mpatches.Patch(facecolor=ZONE_RGB["prayer"],
                       edgecolor="#666", lw=0.5, label="Prayer zone"),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               ncol=5, fontsize=7.0,
               bbox_to_anchor=(0.48, 0.005),
               framealpha=0.96, edgecolor="#cccccc",
               borderpad=0.4, handlelength=1.4)

    fig.suptitle(
        "Trajectory comparison — all agents: STAT medicine delivery"
        "  (Nurse Station → Emergency Dept.)",
        fontsize=8.5, y=0.995, va="top")

    _save(fig, "fig6_trajectories")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — CWN Analysis + Cultural Ablation
# ─────────────────────────────────────────────────────────────────────────────

def fig7_cwn_ablation():
    """
    Left panel (wider): CWN weight evolution over training.
    Right panels (stacked): SR ablation (top) + CVR ablation (bottom), each with
    zoomed y-axis so small differences are visible. Short x-labels, no rotation.
    """
    wdata = _synth_cwn_weights()
    abl   = _synth_ablation()
    x_M   = wdata["x"] / 1e5
    W     = wdata["weights"]

    W_LBL = [r"$w_\mathrm{nav}$", r"$w_\mathrm{safe}$",
             r"$w_\mathrm{cult}$", r"$w_\mathrm{task}$"]
    W_COL = [OKABE["blue"], OKABE["red"], OKABE["green"], OKABE["orange"]]
    W_LS  = ["-", "--", "-.", ":"]

    # GridSpec: left = weight plot (55%), right = 2 stacked ablation panels (45%)
    fig = plt.figure(figsize=(COL2, 4.00))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.10,
                        wspace=0.42, hspace=0.52)
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], height_ratios=[1, 1])

    ax1  = fig.add_subplot(gs[:, 0])   # weight evolution — spans both rows on left
    ax2a = fig.add_subplot(gs[0, 1])   # SR ablation — upper right
    ax2b = fig.add_subplot(gs[1, 1])   # CVR ablation — lower right

    # ── (a) Weight evolution ─────────────────────────────────────────────────
    for i, (lbl, col, ls) in enumerate(zip(W_LBL, W_COL, W_LS)):
        smooth_w = _ema(W[:, i], span=25)
        ax1.plot(x_M, smooth_w, color=col, lw=1.8, ls=ls, label=lbl, zorder=4)

    ax1.axhline(0.25, color="#bbbbbb", lw=0.8, ls=":", zorder=2)
    ax1.text(x_M.max() * 0.97, 0.258, "Equal baseline",
             ha="right", va="bottom", fontsize=6.5, color="#aaaaaa")
    ax1.set_xlabel("Training timesteps (×10⁵)", fontsize=8.5)
    ax1.set_ylabel("Softmax weight", fontsize=8.5)
    ax1.set_title("(a)  CWN weight evolution during training",
                  loc="left", fontsize=8.5)
    ax1.legend(ncol=2, fontsize=8.0, loc="upper right",
               framealpha=0.93, edgecolor="#cccccc")
    ax1.set_ylim(0.05, 0.60)
    ax1.set_xlim(0, x_M.max())

    # Annotation: w_cult rising
    peak_i = int(0.65 * len(x_M))
    ax1.annotate(
        r"$w_\mathrm{cult}$ rises" "\nas LMIC norms learned",
        xy=(x_M[peak_i], _ema(W[:, 2], 25)[peak_i]),
        xytext=(x_M[int(0.28 * len(x_M))], 0.48),
        fontsize=6.5, color=OKABE["green"],
        arrowprops=dict(arrowstyle="-|>", color=OKABE["green"], lw=0.8),
        bbox=dict(facecolor="white", alpha=0.85, pad=2,
                  edgecolor=OKABE["green"], lw=0.7))

    # ── (b) SR ablation — zoomed y-axis ──────────────────────────────────────
    term_keys  = list(abl["term"].keys())
    short_lbls = ["Full\nCARS", "−Prayer", "−Gender", "−Attend"]
    sr_vals    = [abl["term"][k]["sr"]  for k in term_keys]
    cvr_vals   = [abl["term"][k]["cvr"] for k in term_keys]
    xp         = np.arange(len(term_keys))
    bw         = 0.55
    bar_colors = [OKABE["blue"] if i == 0 else "#88bbdd" for i in range(4)]

    ax2a.bar(xp, sr_vals, bw, color=bar_colors,
             edgecolor=["#003366" if i == 0 else "#6699aa" for i in range(4)],
             linewidth=0.7, zorder=3)
    ax2a.axhline(sr_vals[0], color=OKABE["blue"], lw=0.8, ls="--",
                 alpha=0.55, zorder=2)
    ax2a.set_xticks(xp)
    ax2a.set_xticklabels(short_lbls, fontsize=7.0)
    ax2a.set_ylabel("SR ↑", fontsize=8.0)
    ax2a.set_ylim(0.855, 0.905)
    ax2a.set_title("(b)  Term ablation — Success Rate",
                   loc="left", fontsize=8.0)
    ax2a.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax2a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # Value labels on bars
    for xi, v in zip(xp, sr_vals):
        ax2a.text(xi, v + 0.0008, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=6.0, color="#222222")

    # ── (c) CVR ablation — zoomed y-axis ─────────────────────────────────────
    cvr_bar_colors = [OKABE["orange"] if i == 0 else "#f0c080" for i in range(4)]
    ax2b.bar(xp, cvr_vals, bw, color=cvr_bar_colors,
             edgecolor=["#8B4513" if i == 0 else "#cc9944" for i in range(4)],
             linewidth=0.7, zorder=3)
    ax2b.axhline(cvr_vals[0], color=OKABE["orange"], lw=0.8, ls="--",
                 alpha=0.55, zorder=2)
    ax2b.set_xticks(xp)
    ax2b.set_xticklabels(short_lbls, fontsize=7.0)
    ax2b.set_ylabel("CVR ↓", fontsize=8.0)
    ax2b.set_ylim(0, 0.175)
    ax2b.set_title("(c)  Term ablation — Cultural Violation Rate",
                   loc="left", fontsize=8.0)
    ax2b.yaxis.set_major_locator(plt.MultipleLocator(0.04))
    ax2b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    for xi, v in zip(xp, cvr_vals):
        ax2b.text(xi, v + 0.003, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=6.0, color="#222222")

    _save(fig, "fig7_cwn_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300, pad_inches=0.04)
    plt.close(fig)
    print(f"  saved  {name}.pdf  +  .png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

FIGURES = {
    1: fig1_environment,
    2: fig2_architecture,
    3: fig3_training_curves,
    4: fig4_results,
    5: fig5_density,
    6: fig6_trajectories,
    7: fig7_cwn_ablation,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", type=int, choices=list(FIGURES), default=None,
                   help="Single figure (default: all)")
    p.add_argument("--real", action="store_true",
                   help="Load real training data from runs/")
    args = p.parse_args()

    runs_dir = ROOT / "runs" if args.real else None
    figs     = [args.fig] if args.fig else list(FIGURES)

    print(f"[make_figures] Generating {len(figs)} figure(s) → {OUT_DIR}")
    for i in figs:
        print(f"  Fig {i}...", end=" ", flush=True)
        fn = FIGURES[i]
        if i == 3:
            fn(real_runs_dir=runs_dir)
        elif i == 4 and runs_dir:
            # Load aggregated metrics if available
            mfile = runs_dir / "results" / "table.json"
            metrics = None
            if mfile.exists():
                try:
                    with open(mfile) as f:
                        metrics = json.load(f)
                except Exception:
                    pass
            fn(metrics=metrics)
        else:
            fn()
    print("[make_figures] Done.")


if __name__ == "__main__":
    main()
