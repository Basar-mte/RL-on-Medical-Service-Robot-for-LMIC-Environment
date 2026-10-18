"""Exports ALL numerical data of the Gazebo simulation into paper_data/ as plain
CSV (opens in Excel, R, pandas, MATLAB), with a data dictionary (DATA_README.md).

    python3 export_data.py            # re-run any time; it reads only what the runs wrote

Contents
  runs.csv                     one row per run, every campaign (single, realism, repeats,
                               demonstration): configuration, outcome, timing, path,
                               escapes, proxemics, cultural-zone exposure, kinematics,
                               data-quality checks
  repeats_cells.csv            per cell of the repeated campaign: n, successes, success
                               rate + Wilson 95 % interval, median and IQR of each metric
  repeats_escape_tests.csv     escape on vs off per cell: Fisher exact two-sided p
  robot_timeseries.csv.gz      every run's 10 Hz log (run.csv) concatenated, keyed by run_id
  pedestrians/<campaign>/<run_id>.csv.gz
                               every run's walker log (peds.csv: 20 Hz x, y, yaw, walk-cycle
                               time of every walker)
  pedestrian_routes.csv        every walker: level membership, character, speed, loop period
  pedestrian_waypoints.csv     every walker's loop, one row per waypoint
  seed_phases.csv              per level and seed: every walker's start phase and position
  crowd_statistics.csv         robot-free crowd walks: walker spacing, schedule vs simulated
  door_gaps.csv                every doorway's trained opening
  scan_invariance.csv          per tier: the lidar-invariance verification numbers
  parameters.csv               every model constant the runs used, read from the source
"""
from __future__ import annotations
import csv
import glob
import gzip
import json
import math
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "paper_data")
PKG = os.path.join(ROOT, "hospitalnav_deploy")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "hospitalnav_deploy"))
import ped_routes as P                                  # noqa: E402
from zone_metrics import zone_metrics                   # noqa: E402
from analyze_repeats import wilson, fisher_two_sided, q, POLICIES   # noqa: E402

CAMPAIGNS = [("single", os.path.join(ROOT, "results_runs")),
             ("realism", os.path.join(ROOT, "results_runs_realism")),
             ("repeats", os.path.join(ROOT, "results_runs_repeats"))]
N_PEDS = {lvl: len(P.LEVELS[lvl]) for lvl in P.LEVELS}


def fnum(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return "" if v != v else round(v, nd)
    return v


def kinematics(rows):
    """Scalar summaries of a run's 10 Hz log after the patrol starts."""
    if not rows:
        return {}
    t = [float(r["t"]) for r in rows]
    v = [float(r["v_cmd"]) for r in rows]
    moving = [i for i, vv in enumerate(v) if abs(vv) > 0.05]
    t_start = t[moving[0]] if moving else None
    dt = [b - a for a, b in zip(t, t[1:])] + [0.0]
    after = [i for i in range(len(t)) if t_start is not None and t[i] >= t_start]
    stopped = sum(dt[i] for i in after if abs(v[i]) <= 0.02)
    reverse = sum(dt[i] for i in after if v[i] < -0.02)
    fwd = [v[i] for i in after if v[i] > 0.02]
    npeople = [int(float(r["n_people"])) for r in rows if r.get("n_people") not in (None, "")]
    pmin = [float(r["ped_min_m"]) for r in rows if r.get("ped_min_m") not in (None, "", "nan")]
    pmin = [x for x in pmin if x == x]
    key = "sim_abs" if "sim_abs" in rows[0] else "t"
    ts = [float(r[key]) for r in rows]
    gap = max((b - a for a, b in zip(ts, ts[1:])), default=0.0)
    tilt = [float(r["tilt_deg"]) for r in rows if r.get("tilt_deg") not in (None, "")]
    steer = [abs(float(r["steering_cmd"])) for r in rows]
    return {"motion_start_s": t_start, "stopped_s": stopped, "reverse_s": reverse,
            "mean_forward_cmd_mps": (sum(fwd) / len(fwd)) if fwd else None,
            "max_forward_cmd_mps": max(fwd) if fwd else None,
            "mean_abs_steer_rad": sum(steer) / len(steer),
            "mean_tracked_people": (sum(npeople) / len(npeople)) if npeople else None,
            "max_tracked_people": max(npeople) if npeople else None,
            "logged_min_ped_m": min(pmin) if pmin else None,
            "max_tilt_deg": max(tilt) if tilt else None, "max_log_gap_s": gap, "log_rows": len(rows)}


def run_row(campaign, run_dir, r):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "run.csv")))) \
        if os.path.exists(os.path.join(run_dir, "run.csv")) else []
    g = (r.get("goals") or [{}])[-1]
    ped = r.get("pedestrians") or {}
    esc = r.get("escapes") or {}
    zm = zone_metrics(os.path.join(run_dir, "run.csv")) if rows else None
    reached = r.get("outcome") == "reached"
    k = kinematics(rows)
    peds = r.get("peds") or "none"
    row = {
        "campaign": campaign, "run_id": r.get("tag") or os.path.basename(run_dir),
        "tier": r.get("tier"), "policy": r.get("preset"), "pedestrians": peds,
        "n_pedestrians": N_PEDS.get(peds, 0), "ped_seed": r.get("ped_seed", 0),
        "escape": "on" if r.get("escape_enabled") else "off",
        "people_source": r.get("people_source", "lidar"), "route": r.get("route", "direct"),
        "outcome": r.get("outcome"), "reached": int(reached),
        "time_to_goal_s": g.get("seconds") if reached else None,
        "goal_clock_s": g.get("seconds"), "final_distance_m": g.get("final_distance_m"),
        "path_length_m": r.get("path_length_m"),
        "mean_speed_mps": (r.get("path_length_m") / g["seconds"]) if reached and g.get("seconds") else None,
        "stall_escapes": esc.get("stall"), "wedge_escapes": esc.get("wedge"),
        "ped_min_distance_m": ped.get("min_distance_m"), "ped_contacts": ped.get("contacts"),
        "ped_touches": ped.get("touches"), "personal_zone_s": ped.get("personal_zone_s"),
    }
    for z in ("prayer", "male_ward", "female_ward"):
        zz = (zm or {}).get(z) or {}
        row[f"{z}_s"] = zz.get("seconds"); row[f"{z}_m"] = zz.get("metres"); row[f"{z}_max_depth_m"] = zz.get("max_depth_m")
    row["entered_cultural_zone"] = int(bool((zm or {}).get("entered_cultural_zone"))) if zm else None
    row.update(k)
    row["sim_time_s"] = r.get("sim_time_s"); row["wall_time_s"] = r.get("wall_time_s")
    row["sim_to_wall_ratio"] = (r["sim_time_s"] / r["wall_time_s"]) if r.get("sim_time_s") and r.get("wall_time_s") else None
    return row, rows


def collect():
    runs = []                                # (campaign, run_dir, record)
    for camp, root in CAMPAIGNS:
        p = os.path.join(root, "campaign.json")
        if os.path.exists(p):
            recs = None
            for _ in range(5):                   # the running campaign rewrites it after every run
                try:
                    recs = json.load(open(p)); break
                except json.JSONDecodeError:
                    import time; time.sleep(1.0)
            for r in recs or []:
                d = os.path.join(root, r["tag"])
                if os.path.isdir(d):
                    runs.append((camp, d, r))
    for d in sorted(glob.glob(os.path.join(ROOT, "results_figures", "figure_runs", "*"))):
        p = os.path.join(d, "result.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        name = os.path.basename(d)
        # figure-run results carry no configuration: it is in the directory name and run label
        lab = r.get("run_label", "")
        tier = "furnished" if "furnished" in name else "open" if name.startswith("open") else "walled"
        pol = next((x for x in POLICIES if x in name), "fixed_equal")
        peds = "crowd45" if "crowd45" in name else "busy"
        r.update({"tag": f"demo__{name}", "tier": tier, "preset": pol, "peds": peds, "escape_enabled": True,
                  "ped_seed": 0, "run_label": lab})
        runs.append(("demonstration", d, r))
    return runs


def write_csv(path, rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fnum(r.get(k)) for k in cols})
    return len(rows)


def cells_tables(table):
    rep = [r for r in table if r["campaign"] == "repeats"]
    cells = {}
    for r in rep:
        cells.setdefault((r["tier"], r["pedestrians"], r["policy"], r["escape"]), []).append(r)
    out, tests = [], []
    metrics = ["time_to_goal_s", "path_length_m", "ped_contacts", "ped_touches", "ped_min_distance_m",
               "personal_zone_s", "prayer_max_depth_m", "prayer_s", "stall_escapes", "wedge_escapes", "stopped_s"]
    for (tier, peds, pol, esc), rs in sorted(cells.items()):
        k, n = sum(r["reached"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        row = {"tier": tier, "pedestrians": peds, "policy": pol, "escape": esc, "runs": n, "reached": k,
               "success_rate": k / n if n else None, "wilson95_lo": lo, "wilson95_hi": hi,
               "seeds": " ".join(str(r["ped_seed"]) for r in sorted(rs, key=lambda x: x["ped_seed"] or 0))}
        for m in metrics:
            xs = [r[m] for r in rs if r.get(m) is not None]
            row[f"{m}_median"] = q(xs, 0.5) if xs else None
            row[f"{m}_q25"] = q(xs, 0.25) if xs else None
            row[f"{m}_q75"] = q(xs, 0.75) if xs else None
            row[f"{m}_mean"] = (sum(xs) / len(xs)) if xs else None
        out.append(row)
    for (tier, peds, pol, esc), rs in sorted(cells.items()):
        if esc != "on" or (tier, peds, pol, "off") not in cells:
            continue
        off = cells[(tier, peds, pol, "off")]
        a, b = sum(r["reached"] for r in rs), sum(1 - r["reached"] for r in rs)
        c, d = sum(r["reached"] for r in off), sum(1 - r["reached"] for r in off)
        tests.append({"tier": tier, "pedestrians": peds, "policy": pol, "reached_escape_on": a, "runs_escape_on": a + b,
                      "reached_escape_off": c, "runs_escape_off": c + d, "fisher_p_two_sided": fisher_two_sided(a, b, c, d)})
    return out, tests


def environment():
    os.makedirs(OUT, exist_ok=True)
    rows, wps = [], []
    for n, (model, speed, pts) in P.ROUTES.items():
        rows.append({"walker": n, "character": model, "speed_mps": speed, "loop_period_s": P.period(n),
                     "n_waypoints": len(pts), "in_light": int(n in P.LEVELS["light"]),
                     "in_busy": int(n in P.LEVELS["busy"]), "in_crowd45": int(n in P.LEVELS["crowd45"]),
                     "origin": "generated (sim/gen_crowd_routes.py)" if n.startswith("crowd_") else "designed"})
        for i, (x, y) in enumerate(pts):
            wps.append({"walker": n, "index": i, "x_m": x, "y_m": y})
    write_csv(os.path.join(OUT, "pedestrian_routes.csv"), rows)
    write_csv(os.path.join(OUT, "pedestrian_waypoints.csv"), wps)
    ph = []
    for lvl in ("busy", "crowd45"):
        for seed in range(0, 11):
            for n, t0 in P.phases(lvl, seed).items():
                x, y, yaw = P.pose_at(n, t0)
                ph.append({"pedestrians": lvl, "seed": seed, "walker": n, "start_phase_s": t0,
                           "start_x_m": x, "start_y_m": y, "start_yaw_rad": yaw})
    write_csv(os.path.join(OUT, "seed_phases.csv"), ph)
    # crowd statistics of the robot-free walks the stills freeze
    import numpy as np
    cs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results_figures", "crowd_*_*.csv"))):
        tier, lvl = os.path.basename(f)[6:-4].split("_", 1)
        ts, d = P.load_log(f)
        names = list(d)
        X = np.stack([d[n][0] for n in names], 1); Y = np.stack([d[n][1] for n in names], 1)
        SX = np.array([[P.pose_at(n, t)[0] for n in names] for t in ts])
        SY = np.array([[P.pose_at(n, t)[1] for n in names] for t in ts])
        iu = np.triu_indices(len(names), 1)

        def pairs(A, B, thr):
            D = np.hypot(A[:, :, None] - A[:, None, :], B[:, :, None] - B[:, None, :])[:, iu[0], iu[1]]
            return float((D < thr).sum(1).mean())
        S = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)) / np.diff(ts)[:, None]
        cs.append({"tier": tier, "pedestrians": lvl, "walkers": len(names), "duration_s": float(ts[-1] - ts[0]),
                   "touching_pairs_schedule": pairs(SX, SY, 0.44), "touching_pairs_simulated": pairs(X, Y, 0.44),
                   "interpenetrating_pairs_schedule": pairs(SX, SY, 0.30),
                   "interpenetrating_pairs_simulated": pairs(X, Y, 0.30),
                   "mean_walking_speed_mps": float(S.mean()), "walkers_standing_still": float((S < 0.1).sum(1).mean())})
    write_csv(os.path.join(OUT, "crowd_statistics.csv"), cs)
    import gen_worlds as G
    write_csv(os.path.join(OUT, "door_gaps.csv"),
              [{"room": n, "wall_side": s, "wall_coordinate_m": w, "gap_from_m": lo, "gap_to_m": hi,
                "width_m": hi - lo} for n, s, w, lo, hi in G.door_gaps()])
    si = []
    for f in sorted(glob.glob(os.path.join(PKG, "worlds", "scan_invariance_*.json"))):
        r = json.load(open(f))
        si.append({"tier": os.path.basename(f)[16:-5], **{k: v for k, v in r.items() if not isinstance(v, (list, dict))}})
    write_csv(os.path.join(OUT, "scan_invariance.csv"), si)
    # every model constant, read from the source the runs used
    pr = []
    srcs = [("hospitalnav_gz_plugins/src/PedestrianBodies.cc", r"constexpr double (\w+) = ([-\d.e]+);\s*(?://\s*(.*))?"),
            ("hospitalnav_deploy/hospitalnav_deploy/gz_cmd_adapter.py", r"^([A-Z_]+) = ([-\d.e]+|math\.radians\([\d.]+\))\s*(?:#\s*(.*))?$"),
            ("hospitalnav_deploy/hospitalnav_deploy/patrol_sequencer.py", r"^([A-Z_]+) = ([-\d.e]+(?: *\+ *[-\d.e]+)*)\s*(?:#\s*(.*))?$"),
            ("hospitalnav_deploy/hospitalnav_deploy/ped_routes.py", r"^([A-Z_]+) = ([-\d.e]+|\([-\d., ]+\))\s*(?:#\s*(.*))?$")]
    for rel, pat in srcs:
        for line in open(os.path.join(ROOT, rel)):
            m = re.match(pat, line.strip())
            if m:
                pr.append({"source": rel, "name": m.group(1), "value": m.group(2), "comment": (m.group(3) or "").strip()})
    pr += [{"source": "sim/run_campaign.py", "name": "goal_timeout", "value": "300.0",
             "comment": "s of simulation time per goal (campaign runs)"},
            {"source": "hospitalnav_deploy/patrol_sequencer.py", "name": "goal_tol", "value": "0.4",
             "comment": "m: goal reached (training GOAL_TOL)"},
            {"source": "sim/run_campaign.py", "name": "patrol_start_delay", "value": "30.0",
             "comment": "s after launch before the goal clock starts (figure runs: 20)"},
            {"source": "hospitalnav_deploy/patrol_sequencer.py", "name": "contact/touch re-arm", "value": "+0.2",
             "comment": "m: a new contact (touch) event counts only after the distance exceeded the threshold + 0.2"}]
    w = open(os.path.join(PKG, "worlds", "hospital_walled.world")).read()
    for tag in ("max_step_size", "real_time_factor", "real_time_update_rate"):
        m = re.search(rf"<{tag}>([^<]+)</{tag}>", w)
        if m:
            pr.append({"source": "worlds/hospital_walled.world", "name": tag, "value": m.group(1),
                       "comment": "physics (replays use 1/300 s)"})
    write_csv(os.path.join(OUT, "parameters.csv"), pr)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = collect()
    table = []
    ts_path = os.path.join(OUT, "robot_timeseries.csv.gz")
    ts_cols = ["campaign", "run_id", "t", "sim_abs", "x", "y", "yaw", "v_cmd", "steering_cmd", "goal_x", "goal_y",
               "goal_dist", "n_people", "ped_min_m", "z", "tilt_deg", "wall"]
    n_ts = 0
    with gzip.open(ts_path, "wt", newline="") as gz:
        w = csv.DictWriter(gz, fieldnames=ts_cols, extrasaction="ignore")
        w.writeheader()
        for camp, d, r in runs:
            row, rows = run_row(camp, d, r)
            table.append(row)
            for x in rows:
                x = dict(x); x["campaign"] = camp; x["run_id"] = row["run_id"]
                w.writerow(x); n_ts += 1
            src = os.path.join(d, "peds.csv")
            if os.path.exists(src):
                dst = os.path.join(OUT, "pedestrians", camp, row["run_id"] + ".csv.gz")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
                        shutil.copyfileobj(fi, fo)
    n = write_csv(os.path.join(OUT, "runs.csv"), table)
    cells, tests = cells_tables(table)
    write_csv(os.path.join(OUT, "repeats_cells.csv"), cells)
    write_csv(os.path.join(OUT, "repeats_escape_tests.csv"), tests)
    environment()
    by = {}
    for r in table:
        by[r["campaign"]] = by.get(r["campaign"], 0) + 1
    print(f"wrote paper_data/: {n} runs {by}, {n_ts} robot log rows, {len(cells)} repeat cells")


if __name__ == "__main__":
    main()
