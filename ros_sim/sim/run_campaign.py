"""Runs the Gazebo campaign and records one honest outcome per run.

    python3 run_campaign.py                      # the full matrix
    python3 run_campaign.py --presets residual_sf,vanilla --tiers walled
    python3 run_campaign.py --dry-run            # print what would run

MATRIX
  6 policies x 2 tiers x 2 escape settings = 24 runs, every one of them with
  dynamic pedestrians (--peds busy: 8 scripted walkers) unless told otherwise.

    policies  residual_sf      the recommended residual-RL driver
              classical_sf2    tuned social force, no model at all
              fixed_equal      the pure learned CARS policy
              vanilla          culture-blind contrast (takes the prayer-room cut)
              risk_field       the He-2025-style risk-field baseline
              mapless_fixed_equal   no planner - the routing-bottleneck demo
    tiers     walled (Tier-2, enclosed rooms) and open (Tier-1, zones only);
              real (photogrammetry assets, scan-identical) and furnished
              (unmapped furniture + clutter) are the realism tiers, run on
              request with --tiers
    escape    on  = the real MentorPi's stall/wedge recovery is available
              off = policy + shield only, no intervention
    peds      none | light (4 walkers) | busy (8 walkers), see ped_routes.py

CLOCK AND ISOLATION
  Goal budgets are simulation time (the sequencer reads /clock), so a run's
  outcome does not depend on how fast the machine happened to execute it; the
  wall-clock --run-timeout is only a hang guard. Every process of a run carries
  HN_SCOPE=campaign in its environment and clean_sim.sh only kills that scope,
  so a figure capture or probe simulator (scope 'probe', its own gz partition)
  can run alongside without either killing the other.

REPORTING
  Each run yields a binary per-goal outcome (reached / timeout) plus the failure
  mode and the number of escapes used - never a success rate synthesised from
  demonstration runs, per the package's reporting convention. The summary table
  reports runs as "reached in N of M attempts" only when M > 1.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SCRATCH = os.environ.get("HOSPITALNAV_SCRATCH", "/tmp/hospitalnav_campaign")

PRESETS = ["residual_sf", "classical_sf2", "fixed_equal", "vanilla",
           "risk_field", "mapless_fixed_equal"]
TIERS = ["walled", "open"]
ESCAPES = [True, False]

# The run stops at this waypoint: the nurse desk -> emergency room task the
# package documents as its headline demonstration.
FIRST_GOAL = "emergency room"


def clean():
    subprocess.run([os.path.join(HERE, "clean_sim.sh"), "campaign"], capture_output=True)


def run_tag(preset, tier, escape, peds, seed):
    tag = f"{tier}__{preset}__escape-{'on' if escape else 'off'}__peds-{peds}"
    return tag + (f"__seed-{seed:02d}" if seed else "")


def one_run(preset, tier, escape, out_dir, args, peds, seed=0):
    tag = run_tag(preset, tier, escape, peds, seed)
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    csv = os.path.join(d, "run.csv")
    res = os.path.join(d, "result.json")
    log = os.path.join(d, "run.log")
    for f in (csv, res):
        if os.path.exists(f):
            os.remove(f)

    env = dict(os.environ)
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    env["ROS_DOMAIN_ID"] = str(args.domain)
    env["HN_SCOPE"] = "campaign"
    env.pop("GZ_PARTITION", None)          # the campaign owns the default partition

    cmd = [
        "ros2", "launch", "hospitalnav_deploy", "gz_bringup.launch.py",
        f"preset:={preset}", f"tier:={tier}",
        f"escape_enabled:={'true' if escape else 'false'}",
        "gui:=false", "rviz:=false",
        f"patrol_start_delay:={args.start_delay}",
        f"goal_timeout:={args.goal_timeout}",
        f"first_goal_name:={FIRST_GOAL}", f"route:={args.route}",
        f"peds:={peds}", f"ped_seed:={seed}", f"people_source:={args.people_source}",
        f"csv_path:={csv}", f"result_path:={res}", f"run_label:={tag}",
    ]
    if args.slam_params:
        cmd.append(f"slam_params:={args.slam_params}")

    print(f"  -> {tag}", flush=True)
    clean()
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        deadline = t0 + args.run_timeout
        while time.time() < deadline:
            if os.path.exists(res):
                time.sleep(2.0)          # let the file finish being written
                break
            if p.poll() is not None:
                break
            time.sleep(2.0)
        try:
            os.killpg(os.getpgid(p.pid), 15)
            p.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except ProcessLookupError:
                pass
    clean()

    rec = {"tag": tag, "preset": preset, "tier": tier, "peds": peds, "ped_seed": seed,
           "people_source": args.people_source,
           "escape_enabled": escape, "wall_time_s": round(time.time() - t0, 1)}
    if os.path.exists(res):
        with open(res) as f:
            rec.update(json.load(f))
    else:
        rec["outcome"] = "no_result"
        rec["note"] = "run produced no result file (see run.log)"
    print(f"     {rec.get('outcome')}  ({rec['wall_time_s']} s)", flush=True)
    return rec


def summarise(records, out_dir):
    with open(os.path.join(out_dir, "campaign.json"), "w") as f:
        json.dump(records, f, indent=2)

    lines = [
        "# Gazebo campaign results",
        "",
        "Per-run binary outcomes from the gz-sim Harmonic deployment of the",
        "runs_v5 policies on the real MentorPi Ackermann chassis. Each row is ONE",
        "run of the nurse-desk -> emergency-room task (a single goal; the trained",
        "route planner picks the path); `reached` means the robot came within",
        "0.4 m (the training GOAL_TOL) of the destination inside the 300 s",
        "simulation-time budget. Localisation is ground truth; pedestrians",
        "are scripted non-yielding walkers (ped_routes.py); a pedestrian contact is",
        "a ground-truth centre distance below 0.5 m (robot + body radius). These",
        "are demonstration runs: reported as outcomes, not as a success rate.",
        "",
        "| tier | policy | peds | seed | escape | outcome | goals reached | escapes (stall/wedge) "
        "| ped contacts | ped min m | personal-zone s | path m | sim s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        goals = r.get("goals", [])
        nreach = sum(1 for g in goals if g.get("outcome") == "reached")
        esc = r.get("escapes", {})
        ped = r.get("pedestrians", {}) or {}
        pmin = ped.get("min_distance_m")
        lines.append(
            f"| {r['tier']} | {r['preset']} | {r.get('peds','-')} | {r.get('ped_seed', 0)} "
            f"| {'on' if r['escape_enabled'] else 'off'} "
            f"| {r.get('outcome','?')} | {nreach}/{len(goals)} "
            f"| {esc.get('stall','-')}/{esc.get('wedge','-')} "
            f"| {ped.get('contacts','-')} | {'-' if pmin is None else f'{pmin:.2f}'} "
            f"| {ped.get('personal_zone_s','-')} "
            f"| {r.get('path_length_m','-')} | {r.get('sim_time_s','-')} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "campaign.md"), "w") as f:
        f.write(md)
    print("\n" + md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--escapes", default="on,off")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_runs"))
    ap.add_argument("--start-delay", type=float, default=30.0)
    ap.add_argument("--goal-timeout", type=float, default=300.0,
                    help="sim seconds per goal; 300 = ~6x the nominal time of the 37 m direct route")
    ap.add_argument("--route", default="direct", choices=["direct", "hops"])
    ap.add_argument("--run-timeout", type=float, default=1100.0,
                    help="wall-clock hang guard; goal budgets themselves are sim time")
    ap.add_argument("--peds", default="busy",
                    help="pedestrian level(s), comma-separated: none | light | busy | crowd45")
    ap.add_argument("--seeds", default="0",
                    help="pedestrian seeds: '0' (the loop-start schedule), '1-10', or '1,4,7'; seed is the "
                         "OUTER loop, so an interrupted campaign has the same number of seeds in every cell")
    ap.add_argument("--people-source", default="lidar", choices=["lidar", "gt"])
    ap.add_argument("--domain", type=int, default=42)
    ap.add_argument("--slam-params", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    presets = [p.strip() for p in a.presets.split(",") if p.strip()]
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    escapes = [e.strip() == "on" for e in a.escapes.split(",") if e.strip()]

    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    peds_levels = [x.strip() for x in a.peds.split(",") if x.strip()]
    for pl in peds_levels:
        if pl not in ("none", "light", "busy", "crowd45"):
            ap.error(f"unknown pedestrian level '{pl}'")
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        elif part.strip():
            seeds.append(int(part))
    combos = [(p, t, e, pl, sd) for sd in seeds for pl in peds_levels for t in tiers for p in presets for e in escapes]
    print(f"{len(combos)} run(s) -> {out_dir}   (peds={','.join(peds_levels)}, seeds={a.seeds}, "
          f"people_source={a.people_source})\n")
    if a.dry_run:
        for p, t, e, pl, sd in combos:
            print(f"  {run_tag(p, t, e, pl, sd)}")
        return 0

    # Resumable and additive: records already in campaign.json are kept, and a
    # combo whose tag already has an outcome is skipped - so an interrupted
    # campaign continues where it stopped, and a later invocation with other
    # settings (another peds level, the realism tiers) extends the same table
    # instead of overwriting it.
    records = []
    prev = os.path.join(out_dir, "campaign.json")
    if os.path.exists(prev):
        try:
            records = json.load(open(prev))
        except Exception:
            records = []
    done = {r.get("tag") for r in records if r.get("outcome") not in (None, "no_result")}
    t_start = time.time()
    for i, (p, t, e, pl, sd) in enumerate(combos, 1):
        tag = run_tag(p, t, e, pl, sd)
        if tag in done:
            print(f"[{i}/{len(combos)}] {tag}: already done, skipping", flush=True)
            continue
        print(f"[{i}/{len(combos)}] ({(time.time() - t_start) / 3600:.1f} h)", flush=True)
        records = [r for r in records if r.get("tag") != tag]
        records.append(one_run(p, t, e, out_dir, a, pl, sd))
        summarise(records, out_dir)       # checkpoint after every run
    summarise(records, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
