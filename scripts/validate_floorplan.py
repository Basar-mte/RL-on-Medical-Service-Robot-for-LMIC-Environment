"""Floorplan validation — geometry gates BEFORE any training campaign.

Every check is a lesson from the invalidated v2/v3 campaigns (sealed prayer
room, doorway beds, 0.8 m micro-aisles, unreachable task goals):

 1. every task is A*-completable on the ROBOT-inflated grid, using the
    env's own episode setup (real spawn + goal snapping), across 5 seeds
 2. the prayer room is reachable (CVR must stay a measurable risk)
 3. every robot_spawn_point stands on free inflated cells
 4. every room center is reachable from the primary spawn point
 5. env throughput sanity at high density (steps/s)
 6. renders layout_preview.png: V4 additions red, walls black, other
    furniture gray, with the actual A* path for each task overlaid

Exit code 0 = all gates passed.  Usage:
    .venv\\Scripts\\python.exe scripts\\validate_floorplan.py [config.yaml] [out_dir]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.floorplan import FloorPlan  # noqa: E402
from envs.lmic_hospital_env import LMICHospitalNavEnv  # noqa: E402

CFG = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "configs/hospital_v4.yaml")
OUT_DIR = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "runs_v4")

# Labels of V4 crowding additions (drawn red in the preview)
V4_PREFIXES = ("V4 ",)

FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    fp = FloorPlan(CFG)
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    n_obs = len(cfg["static_obstacles"])
    print(f"[validate] {CFG.name}: {n_obs} static obstacles, "
          f"{fp.width_m}x{fp.height_m} m")

    # -- 3. spawn points free on the inflated grid ------------------------
    print("[validate] robot spawn points (inflated grid):")
    for p in fp.robot_spawn_points:
        check(f"spawn {p}", fp.is_walkable_robot(p[0], p[1]))

    # -- 4. all room centers reachable from the primary spawn -------------
    print("[validate] room reachability from primary spawn:")
    p0 = fp.robot_spawn_points[0]
    origin = fp.nearest_walkable(p0[0], p0[1], grid=fp.occupancy_inflated)
    for r_ in fp.rooms:
        c = fp.room_center(r_["id"])
        c = fp.nearest_walkable(c[0], c[1], grid=fp.occupancy_inflated)
        check(f"room {r_['id']}", fp.astar(origin, c) is not None)

    # -- 1. env-level task completability across seeds ---------------------
    print("[validate] task completability via env episode setup:")
    env = LMICHospitalNavEnv(floorplan_path=str(CFG), use_camera=False)
    task_paths = {}
    for task_id in fp.tasks:
        ok_all, lens = True, []
        for seed in range(5):
            env.reset(seed=seed, options={"task": task_id, "density": "medium"})
            path = fp.astar(env._robot_pos, env._task_goal)
            if path is None:
                ok_all = False
                break
            lens.append(sum(np.linalg.norm(path[i + 1] - path[i])
                            for i in range(len(path) - 1)))
            if task_id not in task_paths:
                task_paths[task_id] = (np.array(path),
                                       env._robot_pos.copy(),
                                       env._task_goal.copy())
        check(f"task {task_id}", ok_all,
              f"A* len {np.mean(lens):.1f} m" if ok_all else "NO PATH")

    # -- 2. prayer room specifically (CVR risk must exist) -----------------
    prayer_rooms = [r_ for r_ in fp.rooms if r_.get("type") == "prayer"]
    for r_ in prayer_rooms:
        c = fp.nearest_walkable(*fp.room_center(r_["id"]),
                                grid=fp.occupancy_inflated)
        check(f"prayer room '{r_['id']}' enterable",
              fp.astar(origin, c) is not None)

    # -- 5. throughput sanity at high density ------------------------------
    print("[validate] throughput (high density, random actions):")
    env.reset(seed=0, options={"density": "high"})
    n_steps, t0 = 1000, time.perf_counter()
    for _ in range(n_steps):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset(options={"density": "high"})
    sps = n_steps / (time.perf_counter() - t0)
    check("throughput", sps > 100, f"{sps:.0f} steps/s")

    # -- 6. render the confirmation picture --------------------------------
    fig, ax = plt.subplots(figsize=(13, 9.75))
    zone_colors = {0: "#ffffff", 1: "#f5f5f5", 2: "#fff3d6", 3: "#d9f2e4",
                   4: "#e8f4e8", 5: "#eef3fa", 6: "#fdeaea", 7: "#fbe0d0",
                   8: "#e5f0d3", 9: "#d9e7fb", 10: "#fcdfec"}
    img = np.ones((*fp.zone_map.shape, 3))
    for code, hexc in zone_colors.items():
        rgb = tuple(int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
        img[fp.zone_map == code] = rgb
    ax.imshow(img, origin="lower", extent=[0, fp.width_m, 0, fp.height_m],
              interpolation="nearest")

    for obs in cfg["static_obstacles"]:
        poly = np.array(obs["polygon"])
        x0, y0 = poly[:, 0].min(), poly[:, 1].min()
        w, h = poly[:, 0].max() - x0, poly[:, 1].max() - y0
        is_v4 = any(obs["label"].startswith(p) for p in V4_PREFIXES)
        is_wall = obs["label"].startswith("Wall")
        color = ("#d62728" if is_v4 else
                 "#1a1a1a" if is_wall else "#8a8a8a")
        ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=color,
                                   edgecolor="none", zorder=3))

    task_colors = {"STAT_medicine": "#1f77b4", "nurse_call": "#9467bd",
                   "specimen": "#2ca02c", "restock": "#e377c2",
                   "idle_return": "#7f7f7f"}
    for tid, (path, start, goal) in task_paths.items():
        c = task_colors.get(tid, "black")
        ax.plot(path[:, 0], path[:, 1], color=c, lw=1.8, alpha=0.9,
                zorder=4, label=tid)
        ax.plot(*start, "o", color=c, ms=7, zorder=5)
        ax.plot(*goal, "*", color=c, ms=15, zorder=5,
                markeredgecolor="black", markeredgewidth=0.5)

    for p in fp.robot_spawn_points:
        ax.plot(p[0], p[1], "s", color="#17becf", ms=7, zorder=5,
                markeredgecolor="black", markeredgewidth=0.5)
    for r_ in fp.rooms:
        if r_.get("type") == "walkable":
            continue
        cx, cy = fp.room_center(r_["id"])
        ax.text(cx, cy + 0.6, r_["label"], ha="center", va="center",
                fontsize=8.5, color="#333333", zorder=6, style="italic")

    ax.set_xlim(0, fp.width_m); ax.set_ylim(0, fp.height_m)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{CFG.name} — {n_obs} obstacles (red = V4 extra beds) "
                 f"with per-task A* paths (o = spawn, ★ = goal, "
                 f"cyan ■ = robot spawn points)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    png = OUT_DIR / "layout_preview.png"
    fig.savefig(png, dpi=170)
    print(f"[validate] preview written: {png}")

    print(f"[validate] {'ALL GATES PASSED' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
