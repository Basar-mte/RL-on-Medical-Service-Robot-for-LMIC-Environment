"""Assembles the Gazebo results for the paper into results_summary/gazebo_campaign.md:
per-tier outcome tables (one row per run, binary outcomes), pedestrian proxemics,
the scan-invariance appendix numbers and an index of the figures. Reads only the
files the runs and captures wrote, so it can be re-run at any time.

    python3 make_report.py
"""
from __future__ import annotations
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zone_metrics import zone_metrics   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORLDS = os.path.join(ROOT, "hospitalnav_deploy", "worlds")
OUT = os.path.join(ROOT, "results_summary", "gazebo_campaign.md")
POLICY_NOTE = {
    "residual_sf": "residual RL over the social-force prior (recommended)",
    "classical_sf2": "tuned social force, no model",
    "fixed_equal": "pure learned CARS policy",
    "vanilla": "culture-blind contrast (geo planner)",
    "risk_field": "risk-field baseline (He 2025 style)",
    "mapless_fixed_equal": "no planner - straight-line goal",
}


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return []


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def max_tilt(csv_path):
    """Largest chassis tilt in the run (deg) - None for runs logged before the
    tilt column existed. A robot shoved over by a pedestrian shows > 45."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return max(float(r["tilt_deg"]) for r in rows) if rows and "tilt_deg" in rows[0] else None
    except Exception:
        return None


def grid(records):
    """Outcome of every run, policy x escape - the runs themselves, no rate."""
    pol = [p for p in POLICY_NOTE if any(r.get("preset") == p for r in records)]
    lines = ["| policy | escape on | escape off |", "|---|---|---|"]
    for p in pol:
        cell = {}
        for r in records:
            if r.get("preset") == p:
                cell["on" if r.get("escape_enabled") else "off"] = r.get("outcome", "?")
        lines.append(f"| {p} | {cell.get('on', '-')} | {cell.get('off', '-')} |")
    return "\n".join(lines)


def table(records, run_root):
    lines = ["| tier | policy | peds | escape | outcome | time to goal (sim s) | path (m) | escapes stall/wedge "
             "| prayer zone s / max depth m | gender wards s | ped min (m) | ped contacts (<0.5 m) "
             "| touches (<0.35 m) | personal zone (s) | max tilt (deg) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        csv_path = os.path.join(run_root, r.get("tag", ""), "run.csv")
        zm = zone_metrics(csv_path) if os.path.exists(csv_path) else None
        tilt = max_tilt(csv_path)
        pr = (f"{zm['prayer']['seconds']:.1f} / {zm['prayer']['max_depth_m']:.2f}" if zm else "-")
        gw = (f"{zm['male_ward']['seconds'] + zm['female_ward']['seconds']:.1f}" if zm else "-")
        goals = r.get("goals", [])
        g = goals[-1] if goals else {}
        esc = r.get("escapes", {}) or {}
        ped = r.get("pedestrians", {}) or {}
        t = g.get("seconds") if g.get("outcome") == "reached" else None
        lines.append(
            f"| {r.get('tier')} | {r.get('preset')} | {r.get('peds', '-')} | {'on' if r.get('escape_enabled') else 'off'} "
            f"| **{r.get('outcome', '?')}** | {fmt(t, 1)} | {fmt(r.get('path_length_m'))} "
            f"| {esc.get('stall', '-')}/{esc.get('wedge', '-')} | {pr} | {gw} | {fmt(ped.get('min_distance_m'))} "
            f"| {ped.get('contacts', '-')} | {ped.get('touches', '-')} | {fmt(ped.get('personal_zone_s'), 1)} "
            f"| {fmt(tilt, 1)} |")
    return "\n".join(lines)


def main():
    main_recs = load(os.path.join(ROOT, "results_runs", "campaign.json"))
    realism = load(os.path.join(ROOT, "results_runs_realism", "campaign.json"))
    out = ["# Gazebo (gz-sim Harmonic) campaign - real MentorPi chassis, trained runs_v5 policies", "",
           "Every row is ONE run of the nurse-desk (37, 15) -> emergency-room (5, 7.5) task with the",
           "trained route planner choosing the path on the prebuilt map; `reached` = within 0.4 m",
           "(training GOAL_TOL) inside the 300 s simulation-time budget. Localisation is ground truth",
           "(gz OdometryPublisher), the lidar is the LD19-spec gpu_lidar the LiDAR people tracker",
           "reads. Binary outcomes only - no success rate is synthesised from these demonstration runs.", "",
           "PEDESTRIANS. `busy` = 8 designed walkers, `crowd45` = those 8 + 37 generated ones",
           "(sim/gen_crowd_routes.py: loops planned on the walled map, every room entered through its",
           "doorway). Each walks its own loop under the social force model (Helbing & Molnar 1995;",
           "hospitalnav_gz_plugins PedestrianBodies): people keep their route but give each other and",
           "the robot room - slowing, stopping, stepping around - and never enter a wall. Each has a",
           "70 kg collision body, so contact is physical: the robot cannot drive through a person and a",
           "person shoves the robot, with a bounded force. Ground truth (contacts, distances) is the",
           "simulated walker state, logged per run to peds.csv. `max tilt` checks the robot stayed",
           "upright (> 45 deg would be a knock-down); it is blank for runs logged before the column existed.", "",
           "Prayer-zone depth is the deepest point reached measured from the zone's nearest edge:",
           "the cultural planner routes along the zone's edge (open tier), so a cultural policy",
           "skims it by a few decimetres while a culture-blind one crosses the room (~4 m deep).",
           "Superseded runs are in `results_superseded/` with the reason (2026-10-16 ground-truth",
           "timing; 2026-10-16 ghost pedestrians; 2026-10-17 stiff pedestrian bodies that could fling or",
           "topple the robot) and are not reported here.", ""]
    for peds in ("crowd45", "busy", "none"):
        for tier in ("walled", "open"):
            recs = [r for r in main_recs if r.get("tier") == tier and r.get("peds") == peds]
            if not recs:
                continue
            out += [f"## {tier} tier, pedestrians: {peds}", "", grid(recs), "",
                    table(recs, os.path.join(ROOT, "results_runs")), ""]
    if realism:
        out += ["## Realism tiers (photogrammetry assets; furnished = unmapped clutter)", "",
                table(realism, os.path.join(ROOT, "results_runs_realism")), ""]
    if main_recs or realism:
        out += ["### Policies", ""] + [f"- `{k}`: {v}" for k, v in POLICY_NOTE.items()] + [""]
    out += ["## Scan invariance (what the policy sees vs the trained geometry)", ""]
    for t in ("walled", "real", "open"):
        p = os.path.join(WORLDS, f"scan_invariance_{t}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            out.append(f"- `{t}`: {'PASS' if r.get('passed') else 'FAIL'} - policy 48-ray input differs by at most "
                       f"{r.get('max_policy_48ray_delta_m', 0) * 1000:.2f} mm over {r.get('poses_tested')} poses "
                       f"(raw rays: max {r.get('max_ray_delta_m', 0):.3f} m at {r.get('poses_with_ray_delta', '?')} poses, "
                       f"grazing long-range wall hits)")
    out += ["", "## Figures", ""]
    figs = sorted(glob.glob(os.path.join(ROOT, "results_figures", "*.png")))
    for f in figs:
        out.append(f"- `results_figures/{os.path.basename(f)}`")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(OUT, ROOT), f"({len(main_recs)} main + {len(realism)} realism runs, {len(figs)} figures)")


if __name__ == "__main__":
    main()
