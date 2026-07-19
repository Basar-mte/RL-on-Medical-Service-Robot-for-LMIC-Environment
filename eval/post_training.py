"""
Post-training pipeline: evaluate all trained agents, aggregate results,
and regenerate all figures with real data.

Usage (run after training completes):
  cd lmic-hospitalnav
  python3 eval/post_training.py --runs_dir runs/ --n_episodes 100

This script:
1. Finds the best_model for each agent
2. Runs frozen evaluation (100 episodes)
3. Aggregates mean ± std across seeds
4. Writes results/table.json + results/table.tex
5. Calls make_figures.py --real to regenerate all 7 figures
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

AGENTS = ["cwn", "vanilla", "he2025", "attention", "fixed_equal", "nav_heavy"]
METRICS = ["sr", "cvr", "spl", "mpd", "svr"]


def find_latest_run(runs_dir: Path, agent: str, seed: int) -> Optional[Path]:
    """Find the most recent training run for a given agent+seed."""
    pattern = f"{agent}_seed{seed}_*"
    matches = sorted(runs_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_eval_results(run_dir: Path) -> Optional[Dict]:
    """Load eval_results.json from a run directory."""
    result_file = run_dir / "eval_results.json"
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)
    # Fall back to eval/ subdirectory
    for p in run_dir.glob("**/eval_results.json"):
        with open(p) as f:
            return json.load(f)
    return None


def run_evaluation(run_dir: Path, n_episodes: int = 100) -> Optional[Dict]:
    """Run evaluation on the best model in a run directory."""
    eval_script = Path(__file__).parent / "evaluate.py"

    # EvalCallback saves best_model inside best_model/ subdir
    best_model_dir = run_dir / "best_model"
    if not best_model_dir.exists():
        # Fall back to using the final model zip
        final_zip = run_dir / "final_model.zip"
        if not final_zip.exists():
            print(f"  [skip] No best_model or final_model in {run_dir.name}")
            return None
        best_model_dir = run_dir

    cmd = [
        sys.executable, str(eval_script),
        "--model_dir", str(best_model_dir),
        "--n_episodes", str(n_episodes),
        "--output", "eval_results.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_dir.parent.parent))
    if result.returncode != 0:
        print(f"  [WARN] Eval failed for {run_dir.name}:\n{result.stderr[:400]}")
        return None

    return load_eval_results(run_dir)


def aggregate(runs_dir: Path, n_episodes: int = 100) -> Dict:
    """Evaluate all agents across all seeds and aggregate."""
    all_results = {a: [] for a in AGENTS}

    # Discover all completed runs
    for agent in AGENTS:
        for seed in range(3):
            run_dir = find_latest_run(runs_dir, agent, seed)
            if run_dir is None:
                print(f"  [missing] {agent} seed={seed}")
                continue
            print(f"  Evaluating {agent} seed={seed} from {run_dir.name}...")

            # Try loading cached results first
            res = load_eval_results(run_dir)
            if res is None:
                res = run_evaluation(run_dir, n_episodes)

            if res:
                all_results[agent].append(res)

    # Aggregate
    table = {}
    for agent in AGENTS:
        seed_results = all_results[agent]
        if not seed_results:
            continue
        agg = {}
        for m in METRICS:
            vals = [r.get(m, float("nan")) for r in seed_results]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                agg[m]          = float(np.mean(vals))
                agg[f"{m}_std"] = float(np.std(vals))
                agg[f"{m}_s"]   = float(np.std(vals))  # alias used by figures
        agg["agent"] = agent
        agg["n_seeds"] = len(seed_results)
        table[agent] = agg

    return table


def to_latex(table: Dict) -> str:
    """Generate LaTeX table row per agent."""
    from figures.make_figures import AGENT_LABEL

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Evaluation on LMICHospitalNav-v1 (mean\,$\pm$\,std, 3 seeds, 100 episodes each)}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & SR\,$\uparrow$ & CVR\,$\downarrow$ & SPL\,$\uparrow$ & MPD\,$\uparrow$ \\",
        r"\midrule",
    ]
    agents_order = ["cwn", "attention", "nav_heavy", "fixed_equal", "he2025", "vanilla"]
    for agent in agents_order:
        if agent not in table:
            continue
        d   = table[agent]
        lbl = AGENT_LABEL.get(agent, agent)
        bold_open  = r"\textbf{" if agent == "cwn" else ""
        bold_close = r"}" if agent == "cwn" else ""

        def fmt(m):
            v, s = d.get(m, float("nan")), d.get(f"{m}_std", 0.0)
            if np.isnan(v): return "—"
            return f"{bold_open}{v:.3f}\\,{{\\small $\\pm${s:.3f}}}{bold_close}"

        lines.append(
            f"  {lbl} & {fmt('sr')} & {fmt('cvr')} & {fmt('spl')} & {fmt('mpd')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip evaluation, just aggregate existing results")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = runs_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[post_training] Aggregating evaluation results...")
    table = aggregate(runs_dir, args.n_episodes)

    if not table:
        print("[post_training] No results found. Run training first.")
        sys.exit(1)

    # Save JSON
    table_file = out_dir / "table.json"
    with open(table_file, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[post_training] Saved {table_file}")

    # Save LaTeX
    try:
        latex = to_latex(table)
        tex_file = out_dir / "table.tex"
        tex_file.write_text(latex)
        print(f"[post_training] Saved {tex_file}")
        print("\n" + latex)
    except Exception as e:
        print(f"[warn] LaTeX export failed: {e}")

    # Regenerate figures with real data
    print("\n[post_training] Regenerating figures with real data...")
    fig_script = Path(__file__).parent.parent / "figures" / "make_figures.py"
    subprocess.run([sys.executable, str(fig_script), "--real"], check=False)

    print("\n[post_training] Done.")


if __name__ == "__main__":
    main()
