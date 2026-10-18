"""Evaluate pilot-3 runs on the Tier-2 map exactly like the paper (500 eps,
deterministic, unshielded) — final AND best checkpoints — and compare with the
paper's Tier-2 1M results, overall and on the prayer-shortcut task.

    python experiments/stability_pilot/eval3.py          # evaluate missing, then summarize
    python experiments/stability_pilot/eval3.py --summary
"""
from __future__ import annotations
import collections, glob, json, os, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS = HERE / os.environ.get("PILOT_RUNS", "runs3")
CFG = "configs/hospital_v4.yaml"
AGENTS = ["fixed_equal", "vanilla", "attention"]
SEEDS = [0, 1]
N_EP = 500


def run_dir(a, s):
    hits = sorted(p for p in (RUNS / f"{a}_s{s}").glob(f"{a}_seed{s}_*") if p.is_dir())
    return hits[-1] if hits else None


def targets():
    for a in AGENTS:
        for s in SEEDS:
            rd = run_dir(a, s)
            if rd is None or not (rd / "final_model.zip").exists():
                continue
            # evaluate.py writes to <model_dir>.parent/<output>
            yield a, s, "final", rd, rd.parent / "eval_final500.json"
            if (rd / "best_model").exists():
                yield a, s, "best", rd / "best_model", rd / "eval_best500.json"


def evaluate_missing():
    env = {**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}
    procs = []
    for a, s, tag, model_dir, out in targets():
        if out.exists():
            continue
        cmd = [PY, "eval/evaluate.py", "--model_dir", str(model_dir), "--config", CFG,
               "--n_episodes", str(N_EP), "--output", out.name]
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                      stdout=open(out.with_suffix(".log"), "w", encoding="utf-8"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()


def summarize():
    base = json.load(open(ROOT / "runs_v4" / "results_1M" / "table.json"))
    # paper per-task STAT numbers come from the 2M per-episode records (as in the manuscript)
    paper_stat = {}
    for a in AGENTS:
        recs = [r for f in glob.glob(str(ROOT / "runs_v4" / f"{a}_seed*_*" / "eval_results.json"))
                for r in json.load(open(f))["records"] if r["task_id"] == "STAT_medicine"]
        if recs:
            paper_stat[a] = (np.mean([r["success"] for r in recs]), np.mean([r["cvr"] for r in recs]))
    print(f"{'agent':<12}{'ckpt':<6}{'SR% per seed':<16}{'SR% mean':>9}{'paper 1M':>9}{'CVR':>7}{'paper':>7}"
          f"{'coll%':>7}{'STAT SR%':>9}{'STAT CVR':>9}{'paper STAT':>14}")
    for a in AGENTS:
        for tag in ("final", "best"):
            res = []
            for s in SEEDS:
                rd = run_dir(a, s)
                if rd is None:
                    continue
                f = (rd.parent / "eval_final500.json") if tag == "final" else (rd / "eval_best500.json")
                if f.exists():
                    res.append(json.load(open(f)))
            if not res:
                continue
            recs = [r for e in res for r in e["records"]]
            stat = [r for r in recs if r["task_id"] == "STAT_medicine"]
            ps = paper_stat.get(a, (np.nan, np.nan))
            print(f"{a:<12}{tag:<6}{' / '.join(f'{100*e['sr']:.1f}' for e in res):<16}"
                  f"{100*np.mean([e['sr'] for e in res]):>9.1f}{100*base[a]['sr']:>9.1f}"
                  f"{np.mean([e['cvr'] for e in res]):>7.3f}{base[a]['cvr']:>7.3f}"
                  f"{100*np.mean([r['end']=='collision' for r in recs]):>7.0f}"
                  f"{100*np.mean([r['success'] for r in stat]):>9.1f}{np.mean([r['cvr'] for r in stat]):>9.3f}"
                  f"{f'{100*ps[0]:.0f}% / {ps[1]:.3f}':>14}")


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        evaluate_missing()
    summarize()
