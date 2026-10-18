"""Aggregate shield-weaning results and compare with the 1M Tier-2 baselines.

Writes
  runs/<mode>/results/table.json   per agent (shape of runs*/results/table.json,
                                   needed by eval_shielded as its "base")
  results/comparison.json          mode/agent rows: unshielded SR/CVR/SPL,
                                   shielded SR/CVR (if run), per-task SR,
                                   next to the baseline from runs_v4/results_1M
  results/comparison.md            the same as a table
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "runs"
OUT = HERE / "results"
BASE = ROOT / "runs_v4" / "results_1M"
MATRIX = {"wean": ["fixed_equal", "attention", "vanilla"],
          "fixed": ["fixed_equal"], "naive": ["fixed_equal"]}
SEEDS = [0, 1, 2]
METRICS = ["sr", "cvr", "spl", "svr", "mpd"]


def find_dir(mode, agent, seed):
    hits = sorted(d for d in (RUNS / mode).glob(f"{agent}_seed{seed}_*") if d.is_dir())
    return hits[-1] if hits else None


def load_json(p):
    return json.load(open(p)) if p and p.exists() else None


def main():
    OUT.mkdir(exist_ok=True)
    base_t = load_json(BASE / "table.json") or {}
    base_s = load_json(BASE / "shielded.json") or {}
    rows = []
    for mode, agents in MATRIX.items():
        table = {}
        shielded = load_json(RUNS / mode / "results" / "shielded.json") or {}
        for a in agents:
            res = [r for s in SEEDS if (d := find_dir(mode, a, s))
                   and (r := load_json(d / "eval_results.json"))]
            if not res:
                continue
            row = {"agent": a, "n_seeds": len(res)}
            for m in METRICS:
                v = [r[m] for r in res]
                row[m], row[f"{m}_std"] = float(np.mean(v)), float(np.std(v))
            table[a] = row
            recs = [e for r in res for e in r["records"]]
            per_task = {t: float(np.mean([e["success"] for e in recs if e["task_id"] == t]))
                        for t in sorted({e["task_id"] for e in recs})}
            ends = {k: float(np.mean([e["end"] == k for e in recs]))
                    for k in ("success", "collision", "timeout")}
            sh = shielded.get(a + "+shield", {})
            b, bs = base_t.get(a, {}), base_s.get(a + "+shield", {})
            rows.append(dict(
                mode=mode, agent=a, n_seeds=len(res),
                sr=row["sr"], sr_std=row["sr_std"], cvr=row["cvr"], spl=row["spl"],
                collision=ends["collision"], timeout=ends["timeout"], per_task=per_task,
                sr_shield=sh.get("sr"), cvr_shield=sh.get("cvr"),
                base_sr=b.get("sr"), base_cvr=b.get("cvr"), base_sr_shield=bs.get("sr")))
        if table:
            (RUNS / mode / "results").mkdir(parents=True, exist_ok=True)
            json.dump(table, open(RUNS / mode / "results" / "table.json", "w"), indent=2)

    json.dump(rows, open(OUT / "comparison.json", "w"), indent=2)

    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"

    def f3(x):
        return "-" if x is None else f"{x:.3f}"

    md = ["| mode | agent | SR % (no shield) | baseline SR % | CVR | baseline CVR "
          "| SR % +shield | baseline +shield | collision % | STAT SR % |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['mode']} | {r['agent']} | {pct(r['sr'])} ± {pct(r['sr_std'])} | "
                  f"{pct(r['base_sr'])} | {f3(r['cvr'])} | {f3(r['base_cvr'])} | "
                  f"{pct(r['sr_shield'])} | {pct(r['base_sr_shield'])} | "
                  f"{pct(r['collision'])} | {pct(r['per_task'].get('STAT_medicine'))} |")
    (OUT / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
