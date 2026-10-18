"""Data-quality repair for a campaign directory: a run without an outcome
(aborted - cut off by the wall-clock guard, e.g. when the network dropped and
gz transport discovery broke), or a run whose log has a gap
> 1 s (the pose was unavailable - measured once when a heavy render shared the
machine) or whose timeout was recorded after more than 301 s is moved to
<campaign>/_excluded_quality/ with the reason, and its record is removed from
campaign.json, so the next run_campaign.py invocation re-runs it.

    python3 quality_repair.py ../results_runs_repeats
"""
import csv, json, os, shutil, sys

root = os.path.abspath(sys.argv[1])
p = os.path.join(root, "campaign.json")
recs = json.load(open(p))
keep, bad = [], []
for r in recs:
    d = os.path.join(root, r["tag"])
    why = None
    try:
        rows = list(csv.DictReader(open(os.path.join(d, "run.csv"))))
        key = "sim_abs" if rows and "sim_abs" in rows[0] else "t"   # logs before 2026-10-16 have no sim_abs
        ts = [float(x[key]) for x in rows]
        gap = max((b - a for a, b in zip(ts, ts[1:])), default=0.0)
        if gap > 1.0:
            why = f"log gap {gap:.1f} s"
    except Exception as e:
        why = f"unreadable log ({e})"
    if r.get("outcome") not in ("reached", "timeout"):
        why = (why + "; " if why else "") + f"no outcome ({r.get('outcome')}: the run was cut off, e.g. by the wall-clock guard)"
    g = (r.get("goals") or [{}])[-1]
    if r.get("outcome") == "timeout" and (g.get("seconds") or 0) > 301.0:
        why = (why + "; " if why else "") + f"timeout recorded at {g.get('seconds')} s"
    if why:
        bad.append((r["tag"], why))
    else:
        keep.append(r)
ex = os.path.join(root, "_excluded_quality")
for tag, why in bad:
    os.makedirs(ex, exist_ok=True)
    dst = os.path.join(ex, tag)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if os.path.isdir(os.path.join(root, tag)):
        shutil.move(os.path.join(root, tag), dst)
    with open(os.path.join(ex, "REASONS.txt"), "a") as f:
        f.write(f"{tag}: {why}\n")
json.dump(keep, open(p, "w"), indent=2)
print(f"{len(bad)} run(s) set aside for re-running: {bad}")
