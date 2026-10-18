"""
Phase 6.2 — Aggregate multi-seed results and compute statistics.

Reads eval_results.json from all seed runs, computes mean ± std,
and produces a LaTeX-ready results table.

Usage:
  python eval/aggregate.py --runs_dir runs/ --output runs/results/table.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


METRICS = ["sr", "cvr", "spl", "mpd", "svr", "ct_s"]
METRIC_LABELS = {
    "sr":  "SR (↑)",
    "cvr": "CVR (↓)",
    "spl": "SPL (↑)",
    "mpd": "MPD (↑)",
    "svr": "SVR (↓)",
    "ct_s":"CT (s) (↓)",
}

AGENTS = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]
AGENT_LABELS = {
    "cwn":        "CWN (ours)",
    "fixed_equal":"B1: Equal Weights",
    "nav_heavy":  "B2: Nav-Heavy",
    "vanilla":    "B3: Vanilla SAC",
    "he2025":     "B4: He et al. (2025)",
    "attention":  "B5: Attn. Weights",
}


def load_runs(runs_dir: Path) -> Dict[str, List[Dict]]:
    """
    Returns dict: agent_name → list of summary dicts (one per seed).
    """
    data: Dict[str, List] = {a: [] for a in AGENTS}
    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        name = run_dir.name  # e.g. "cwn_seed0_1234567890"
        agent = name.split("_seed")[0] if "_seed" in name else None
        if agent not in AGENTS:
            continue
        result_file = run_dir / "best_model" / "eval_results.json"
        if not result_file.exists():
            result_file = run_dir / "eval_results.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            blob = json.load(f)
        summary = blob.get("summary", blob)
        data[agent].append(summary)
    return data


def aggregate(data: Dict[str, List[Dict]]) -> pd.DataFrame:
    rows = []
    for agent in AGENTS:
        seeds = data[agent]
        if not seeds:
            rows.append({"agent": AGENT_LABELS[agent], **{m: "—" for m in METRICS}})
            continue
        row = {"agent": AGENT_LABELS[agent]}
        for m in METRICS:
            vals = [s[m] for s in seeds if m in s]
            if not vals:
                row[m] = "—"
            elif len(vals) == 1:
                row[m] = f"{vals[0]:.3f}"
            else:
                row[m] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame) -> str:
    # Hand-built rather than df.to_latex(): pandas >=2.1 routes to_latex()
    # through its Styler, which requires jinja2>=3.1.2 — an unrelated
    # environment dependency this simple table doesn't need to incur.
    col_map = {m: METRIC_LABELS[m] for m in METRICS}
    df_latex = df.rename(columns={"agent": "Method", **col_map})
    headers = list(df_latex.columns)
    col_spec = "l" + "c" * (len(headers) - 1)

    lines = [
        r"\begin{table}", r"\centering",
        r"\caption{Main results on LMICHospitalNav-v1 (mean \(\pm\) std, 3 seeds).}",
        r"\label{tab:main_results}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule",
    ]
    for _, row in df_latex.iterrows():
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    # Previously "results/table.json" (repo root) — a separate location from
    # what eval/post_training.py writes and figures/make_figures.py reads
    # ("runs/results/table.json"). Defaulting to the same path here avoids
    # the two aggregation tools silently diverging.
    p.add_argument("--output", default="runs/results/table.json")
    args = p.parse_args()

    runs_dir = Path(__file__).parent.parent / args.runs_dir
    data = load_runs(runs_dir)
    df = aggregate(data)

    out_path = Path(__file__).parent.parent / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    latex = to_latex(df)
    tex_path = out_path.with_suffix(".tex")
    with open(tex_path, "w") as f:
        f.write(latex)

    print(df.to_string(index=False))
    print(f"\n[aggregate] Saved to {out_path} and {tex_path}")
