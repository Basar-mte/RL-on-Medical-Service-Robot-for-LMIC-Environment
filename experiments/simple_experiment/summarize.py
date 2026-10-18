"""Aggregate the simplified-experiment evaluations into runs/results/.

Reads each run's cached eval JSONs (never re-evaluates, never touches the
paper figures) and writes:
  table.json          per agent: mean/std over seeds of sr, cvr, spl, svr, mpd
                      (same shape as runs*/results/table.json)
  per_task.json       per agent x task: SR / CVR pooled over seeds
  density_sweep.json  per agent: SR / CVR at 10, 10-20, 20 pedestrians
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "results"
AGENTS = ["fixed_equal", "vanilla", "attention", "cwn", "nav_heavy", "he2025"]
SEEDS = [0, 1, 2]
METRICS = ["sr", "cvr", "spl", "svr", "mpd"]
SWEEP = {"low (10)": "eval_low.json", "medium (10-20)": "eval_results.json",
         "high (20)": "eval_high.json"}


def find_dir(agent, seed):
    hits = sorted(d for d in RUNS.glob(f"{agent}_seed{seed}_*") if d.is_dir())
    return hits[-1] if hits else None


def load(agent, seed, name):
    d = find_dir(agent, seed)
    f = d / name if d else None
    return json.load(open(f)) if f and f.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, per_task, sweep = {}, {}, {}
    for a in AGENTS:
        res = [r for s in SEEDS if (r := load(a, s, "eval_results.json"))]
        if not res:
            continue
        row = {"agent": a, "n_seeds": len(res)}
        for m in METRICS:
            v = [r[m] for r in res]
            row[m], row[f"{m}_std"] = float(np.mean(v)), float(np.std(v))
        table[a] = row

        recs = [e for r in res for e in r["records"]]
        per_task[a] = {}
        for t in sorted({e["task_id"] for e in recs}):
            te = [e for e in recs if e["task_id"] == t]
            per_task[a][t] = dict(sr=float(np.mean([e["success"] for e in te])),
                                  cvr=float(np.mean([e["cvr"] for e in te])), n=len(te))

        sweep[a] = {}
        for label, fname in SWEEP.items():
            rs = [r for s in SEEDS if (r := load(a, s, fname))]
            if rs:
                sweep[a][label] = dict(sr=float(np.mean([r["sr"] for r in rs])),
                                       cvr=float(np.mean([r["cvr"] for r in rs])),
                                       n_seeds=len(rs))

    for name, obj in (("table.json", table), ("per_task.json", per_task),
                      ("density_sweep.json", sweep)):
        json.dump(obj, open(OUT / name, "w"), indent=2)

    print(f"{'agent':<12}{'SR %':>14}{'CVR':>16}{'SPL':>8}")
    for a, r in sorted(table.items(), key=lambda kv: -kv[1]["sr"]):
        print(f"{a:<12}{100*r['sr']:>8.1f} ±{100*r['sr_std']:>4.1f}"
              f"{r['cvr']:>10.3f} ±{r['cvr_std']:.3f}{r['spl']:>8.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
