"""Builds paper_figures/: labelled multi-panel collages (PDF + 600-dpi PNG),
the selected full-resolution single images, the trajectory figures and the
demo video, with captions and provenance in paper_figures/README.md.

    python3 make_paper_figures.py

Callouts in the Gazebo panels are placed by projecting known world positions
(robot, signs, furniture, the pedestrians at the render's simulation time)
through each camera's recorded pose and lens, so every arrow points at the
object it names rather than at a hand-picked pixel.
"""
from __future__ import annotations
import glob
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
HI = os.path.join(ROOT, "results_figures", "hires")
FR = os.path.join(ROOT, "results_figures", "figure_runs")
OUT = os.path.join(ROOT, "paper_figures")
sys.path.insert(0, os.path.join(ROOT, "hospitalnav_deploy", "hospitalnav_deploy"))
import ped_routes as P   # noqa: E402

DPI = 600
FIG_W = 7.16                       # IEEE double-column width (in)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
STROKE = [pe.withStroke(linewidth=2.2, foreground="white")]


# ------------------------------------------------------------- image access --
def load(path, max_w=None):
    im = Image.open(path).convert("RGB")
    if max_w and im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    return im


def meta(png):
    j = png[:-4] + ".json"
    return json.load(open(j)) if os.path.exists(j) else None


def project(m, X, size):
    """World point X -> pixel (u, v) in an image of `size`, or None if behind the camera."""
    (cx, cy, cz), (_r, p, y) = m["pose_xyz"], m["pose_rpy"]
    W, H = size
    d = np.array(X, float) - np.array([cx, cy, cz])
    cyw, syw, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
    Rz = np.array([[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    c = (Rz @ Ry).T @ d                       # camera frame: x forward, y left, z up
    if c[0] <= 0.05:
        return None
    f = (W / 2) / math.tan(m["horizontal_fov_rad"] / 2)
    u, v = W / 2 - f * c[1] / c[0], H / 2 - f * c[2] / c[0]
    return (u, v) if 0 <= u < W and 0 <= v < H else None


def pedestrians_in_view(m, size, level=None):
    """Walkers visible in a still, nearest first. A still rendered from a crowd
    log (capture_figures --peds-log) holds the walkers at that log's instant;
    an older one used the raw schedule at the frame's sim time."""
    level = level or m.get("peds") or "busy"
    if level == "none" or level not in P.LEVELS:
        return []
    if m.get("peds_log") and os.path.exists(m["peds_log"]):
        log = P.load_log(m["peds_log"])
        where = lambda n: P.logged_pose(log, n, m["peds_at"])
    elif m.get("sim_stamp_s") is not None:
        where = lambda n: P.pose_at(n, m["sim_stamp_s"])
    else:
        return []
    out = []
    for n in P.LEVELS[level]:
        x, y, _ = where(n)
        uv = project(m, (x, y, 1.0), size)
        if uv:
            dist = math.dist((x, y), m["pose_xyz"][:2])
            out.append((dist, uv, n))
    return sorted(out)


# ----------------------------------------------------------------- drawing --
def label(ax, letter, text):
    ax.text(0.012, 0.975, rf"$\bf{{({letter})}}$ {text}", transform=ax.transAxes, va="top", ha="left",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.9))


def callout(ax, uv, text, dx, dy, size):
    """Arrow from a text box offset (dx, dy) in image fractions to pixel uv."""
    W, H = size
    tx = min(max(uv[0] + dx * W, 0.04 * W), 0.96 * W)
    ty = min(max(uv[1] + dy * H, 0.12 * H), 0.95 * H)
    ax.annotate(text, xy=uv, xytext=(tx, ty), fontsize=6.2, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333", lw=0.4, alpha=0.95),
                arrowprops=dict(arrowstyle="-|>", color="#111", lw=0.8, shrinkA=2, shrinkB=1.5,
                                path_effects=STROKE), zorder=5)


def show(ax, im):
    ax.imshow(np.asarray(im), interpolation="lanczos")
    ax.set_xlim(0, im.width); ax.set_ylim(im.height, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#444")


def grid_figure(rows, cols, aspect, gap=0.012, extra_bottom=0.0):
    """A figure of rows x cols panels of the given width/height aspect, FIG_W wide."""
    pw = (1 - gap * (cols + 1)) / cols
    ph_in = pw * FIG_W / aspect
    H_in = rows * ph_in + (rows + 1) * gap * FIG_W + extra_bottom
    fig = plt.figure(figsize=(FIG_W, H_in))
    axs = []
    for r in range(rows):
        for c in range(cols):
            x0 = gap + c * (pw + gap)
            y0 = 1 - (gap * FIG_W + (r + 1) * ph_in + r * gap * FIG_W) / H_in
            axs.append(fig.add_axes([x0, y0, pw, ph_in / H_in]))
    return fig, axs


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=DPI)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=DPI)
    plt.close(fig)
    print("wrote", f"paper_figures/{name}.pdf/.png")


def crop_aspect(im, aspect):
    w, h = im.size
    if w / h > aspect:
        nw = round(h * aspect); x0 = (w - nw) // 2
        return im.crop((x0, 0, x0 + nw, h))
    nh = round(w / aspect); y0 = (h - nh) // 2
    return im.crop((0, y0, w, y0 + nh))


def pad_aspect(im, aspect, fill=(255, 255, 255)):
    """Letterbox to `aspect` (white), keeping the whole image."""
    w, h = im.size
    if abs(w / h - aspect) < 1e-3:
        return im
    W, H = (round(h * aspect), h) if w / h < aspect else (w, round(w / aspect))
    out = Image.new("RGB", (W, H), fill)
    out.paste(im.convert("RGB"), ((W - w) // 2, (H - h) // 2))
    return out


def rviz_floor(im, scale=64.0, margin=1.0):
    """Crop a 4K top-down RViz still to the 40 x 30 m floor (+margin): the
    view is centred on (20, 15) at `scale` px/m in the render area."""
    W, H = im.size
    x0 = W / 2 + (-margin - 20) * scale; x1 = W / 2 + (40 + margin - 20) * scale
    y0 = H / 2 - (30 + margin - 15) * scale; y1 = H / 2 - (-margin - 15) * scale
    return im.crop((max(0, round(x0)), max(0, round(y0)), min(W, round(x1)), min(H, round(y1))))


# ----------------------------------------------------------------- figures --
def fig_environment():
    shots = [("oblique", "Overview of the furnished ward floor"),
             ("corridor", "Main corridor with walking pedestrians"),
             ("nursedesk", "Nurse station at the start of the task"),
             ("ward", "General Ward 1: occupied bed bays"),
             ("prayer", "Prayer room (culturally restricted zone)"),
             ("emergency", "Emergency room, the navigation goal")]
    fig, axs = grid_figure(3, 2, 16 / 9)
    for k, (ax, (shot, text)) in enumerate(zip(axs, shots)):
        png = os.path.join(HI, f"gazebo_furnished_{shot}.png")
        im = load(png, 2600)
        m = meta(png)
        show(ax, im)
        label(ax, "abcdef"[k], text)
        if not m:
            continue
        s = im.size
        sc = im.width / m["resolution"][0]
        pj = lambda X: (lambda uv: (uv[0] * sc, uv[1] * sc) if uv else None)(project(m, X, m["resolution"]))
        if shot == "corridor":
            for X, t, dx, dy in (((33.0, 14.9, 0.12), "MentorPi robot", 0.13, 0.12),
                                 ((33.5, 16.86, 2.1), "cultural-zone sign", -0.23, -0.07),
                                 ((36.3, 15.95, 1.0), "nurse station", 0.07, 0.19)):
                uv = pj(X)
                if uv:
                    callout(ax, uv, t, dx, dy, s)
            peds = pedestrians_in_view(m, m["resolution"])
            if peds:
                uv = peds[0][1]
                callout(ax, (uv[0] * sc, uv[1] * sc), "walking pedestrian\n(simulated walker)", -0.26, -0.12, s)
        if shot == "prayer":
            uv = pj((13.1, 6.6, 0.03))
            if uv:
                callout(ax, uv, "prayer mats", 0.18, -0.12, s)
        if shot == "emergency":
            for X, t, dx, dy in (((7.6, 3.3, 1.0), "X-ray unit", 0.0, -0.22),
                                 ((2.25, 5.0, 0.7), "trolley bed", 0.10, -0.22)):
                uv = pj(X)
                if uv:
                    callout(ax, uv, t, dx, dy, s)
        if shot == "ward":
            for X, t, dx, dy in (((14.05, 22.0, 0.75), "occupied patient bed", 0.20, -0.20),
                                 ((13.45, 23.0, 0.55), "bedside cabinet", 0.12, 0.20),
                                 ((14.35, 25.0, 0.45), "visitor chair", 0.22, 0.12)):
                uv = pj(X)
                if uv:
                    callout(ax, uv, t, dx, dy, s)
    save(fig, "fig1_environment")


def fig_tiers():
    tiers = [("reference", "Trained geometry: what the LD19 lidar sees"),
             ("walled", "Walled ward (Tier 2), as the cameras see it"),
             ("open", "Open ward (Tier 1): rooms are semantic zones only"),
             ("furnished", "Furnished: 97 unmapped props with collision")]
    fig, axs = grid_figure(2, 2, 4 / 3)
    for k, (ax, (t, text)) in enumerate(zip(axs, tiers)):
        show(ax, load(os.path.join(HI, f"gazebo_{t}_floorplan.png"), 2400))
        label(ax, "abcd"[k], text)
    save(fig, "fig2_world_tiers")


def fig_robot():
    fig, axs = grid_figure(2, 2, 16 / 9)
    items = [("robot_closeup", "MentorPi Ackermann robot"),
             ("robot_eye", "Robot's-eye view down the corridor"),
             ("chase", "Chase view: the robot among pedestrians")]
    for k, (shot, text) in enumerate(items):
        png = os.path.join(HI, f"gazebo_furnished_{shot}.png")
        im = load(png, 2600); m = meta(png)
        show(axs[k], im); label(axs[k], "abc"[k], text)
        if m and shot == "robot_closeup":
            sc = im.width / m["resolution"][0]
            for X, t, dx, dy in (((33.0096, 14.9, 0.165), "LD19 lidar (scan plane 0.154 m)", -0.18, -0.30),
                                 ((32.936, 14.9, 0.112), "depth camera", -0.22, 0.10),
                                 ((32.933, 14.83, 0.03), "steered front wheels", 0.10, 0.28)):
                uv = project(m, X, m["resolution"])
                if uv:
                    callout(axs[k], (uv[0] * sc, uv[1] * sc), t, dx, dy, im.size)
        if m and shot == "chase":
            sc = im.width / m["resolution"][0]
            uv = project(m, (33.0, 14.9, 0.12), m["resolution"])
            if uv:
                callout(axs[k], (uv[0] * sc, uv[1] * sc), "MentorPi", 0.13, -0.13, im.size)
    local = os.path.join(FR, "walled_fixed_equal_local", "rviz_local.png")
    if os.path.exists(local):
        show(axs[3], crop_aspect(load(local, 2600), 16 / 9))
        label(axs[3], "d", "RViz, around the robot: LD19 scan on the map")
    save(fig, "fig3_robot_and_sensing")


def rviz_legend(fig):
    h = [Line2D([], [], color="#d7191c", lw=2.2, label="robot path"),
         Line2D([], [], color="#1464dc", lw=2.2, label="cultural route (planner)"),
         Line2D([], [], ls="none", marker="o", ms=3, color="#0096be", label="LD19 scan"),
         Line2D([], [], ls="none", marker="o", ms=6, color="#f08c28", label="pedestrian (ground truth)"),
         Line2D([], [], ls="none", marker="o", ms=5, color="#2ca02c", label="start"),
         Line2D([], [], ls="none", marker="o", ms=5, color="#1a59d9", label="goal"),
         Patch(fc="#6f8079", ec="none", label="unknown (SLAM)")]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.6, columnspacing=1.0)


def run_facts(run):
    """(outcome text, prayer-room depth m) of a figure run, from its own logs."""
    sys.path.insert(0, HERE)
    from zone_metrics import zone_metrics
    d = os.path.join(FR, run)
    res = json.load(open(os.path.join(d, "result.json")))
    g = (res.get("goals") or [{}])[-1]
    out = f"reached in {g['seconds']:.0f} s" if g.get("outcome") == "reached" else str(g.get("outcome"))
    depth = zone_metrics(os.path.join(d, "run.csv"))["prayer"]["max_depth_m"]
    return out, depth


def fig_rviz():
    w_out, _ = run_facts("walled_fixed_equal_video")
    v_out, v_dep = run_facts("open_vanilla")
    f_out, f_dep = run_facts("open_fixed_equal")
    panels = [(os.path.join(FR, "walled_fixed_equal_video", "rviz_topdown.png"),
               f"Walled, fixed_equal: route, path, scan; {w_out}"),
              (os.path.join(FR, "walled_fixed_equal_video", "rviz_slam_final.png"),
               "Live SLAM map built during the same run"),
              (os.path.join(FR, "open_vanilla", "rviz_topdown.png"),
               f"Open, vanilla (culture-blind): {v_dep:.1f} m into the prayer room"),
              (os.path.join(FR, "open_fixed_equal", "rviz_topdown.png"),
               f"Open, fixed_equal (cultural): skirts the prayer room, {f_dep:.1f} m")]
    fig, axs = grid_figure(2, 2, 42 / 32, extra_bottom=0.22)
    for k, (ax, (png, text)) in enumerate(zip(axs, panels)):
        if os.path.exists(png):
            show(ax, rviz_floor(load(png)))
        label(ax, "abcd"[k], text)
    rviz_legend(fig)
    save(fig, "fig4_rviz_navigation")


def fig_video_frames(video, name="fig7_video_frames"):
    if not os.path.exists(video):
        return
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    stamps = [0.15, 0.45, 0.8]
    import re
    info = subprocess.run([ff, "-i", video], capture_output=True, text=True).stderr
    hh, mm, ss = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    total = int(hh) * 3600 + int(mm) * 60 + float(ss)
    fig, axs = grid_figure(1, 3, 16 / 9)
    for k, (ax, f) in enumerate(zip(axs, stamps)):
        t = f * total
        tmp = os.path.join(OUT, f".vf{k}.png")
        subprocess.run([ff, "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", tmp])
        show(ax, load(tmp, 1920)); os.remove(tmp)
        label(ax, "abc"[k], f"video, t = {t:.0f} s")
    save(fig, name)


# the 45-walker demonstration shown in fig8 / fig11: the FIRST attempt (both attempts are
# kept and their outcomes documented in README.md)
CROWD_RUN = os.environ.get("HN_CROWD_RUN", "walled_fixed_equal_crowd45_video")


def fig_crowd():
    """45 walking pedestrians: what the cameras see, and the run in RViz."""
    run = os.path.join(FR, CROWD_RUN)
    res = json.load(open(os.path.join(run, "result.json"))) if os.path.exists(os.path.join(run, "result.json")) else {}
    g = (res.get("goals") or [{}])[-1]
    outcome = (f"reached in {g.get('seconds', 0):.0f} s" if g.get("outcome") == "reached"
               else f"{g.get('outcome', '?')} (300 s budget)")
    panels = [(os.path.join(HI, "gazebo_walled_corridor_crowd45.png"), "45 walking pedestrians (social-force walkers)"),
              (os.path.join(HI, "gazebo_walled_oblique_crowd45.png"), "The ward with 45 walkers, from above"),
              (os.path.join(HI, "gazebo_walled_chase_crowd45.png"), "Chase view: the MentorPi in the crowd"),
              (os.path.join(run, "rviz_topdown.png"), f"RViz: one 45-walker run, fixed_equal, {outcome}")]
    fig, axs = grid_figure(2, 2, 16 / 9)
    for k, (ax, (png, text)) in enumerate(zip(axs, panels)):
        if not os.path.exists(png):
            continue
        if "rviz" in os.path.basename(png):
            im = pad_aspect(rviz_floor(load(png)), 16 / 9)      # crop at the render's own scale
        else:
            im = load(png, 2600)
        show(ax, im); label(ax, "abcd"[k], text)
        m = meta(png)
        if m and k == 2:
            sc = im.width / m["resolution"][0]
            uv = project(m, (33.0, 14.9, 0.12), m["resolution"])
            if uv:
                callout(ax, (uv[0] * sc, uv[1] * sc), "MentorPi", 0.13, -0.13, im.size)
    save(fig, "fig8_crowd45")


def singles():
    d = os.path.join(OUT, "singles")
    os.makedirs(d, exist_ok=True)
    picks = {f"gazebo_{t}_{s}.png": f"gazebo_{t}_{s}.png"
             for t in ("furnished",) for s in ("oblique", "corridor", "nursedesk", "ward", "prayer", "emergency",
                                               "chase", "robot_closeup", "robot_eye", "nursedesk_east", "floorplan")}
    picks.update({f"gazebo_{t}_{s}.png": f"gazebo_{t}_{s}.png" for t in ("walled", "open", "reference")
                  for s in ("floorplan", "oblique", "corridor", "chase", "ward", "door_ward", "door_er")})
    picks.update({f"gazebo_walled_{s}_crowd45.png": f"gazebo_walled_{s}_crowd45.png"
                  for s in ("floorplan", "oblique", "corridor", "chase")})
    n = 0
    for src, dst in picks.items():
        p = os.path.join(HI, src)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(d, dst)); n += 1
    for run in ("walled_fixed_equal_video", "walled_fixed_equal_crowd45_video", "walled_fixed_equal_crowd45_video2",
                "open_vanilla", "open_fixed_equal", "furnished_fixed_equal"):
        for p in glob.glob(os.path.join(FR, run, "rviz_*.png")):
            shutil.copy2(p, os.path.join(d, f"rviz_{run}_{os.path.basename(p)[5:]}")); n += 1
    print(f"copied {n} full-resolution singles to paper_figures/singles/")


def trajectories():
    src = os.path.join(ROOT, "results_figures")
    for name, out in (("paths_open_peds-busy_compare", "fig5_trajectories_open_ward"),
                      ("paths_walled_peds-busy_compare", "fig6_trajectories_walled_ward"),
                      ("paths_walled_peds-crowd45_compare", "fig9_trajectories_walled_crowd45"),
                      ("paths_open_peds-crowd45_compare", "fig10_trajectories_open_crowd45")):
        for ext in (".pdf", ".png"):
            p = os.path.join(src, name + ext)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(OUT, out + ext))
        print("copied", out)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    video = os.path.join(OUT, "video", "hospitalnav_demo.mp4")
    for f in (fig_environment, fig_tiers, fig_robot, fig_rviz, fig_crowd, trajectories, singles):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: {e}")
    fig_video_frames(video)
    fig_video_frames(os.environ.get("HN_CROWD_VIDEO", os.path.join(OUT, "video", "hospitalnav_crowd45.mp4")),
                     "fig11_crowd45_video_frames")
