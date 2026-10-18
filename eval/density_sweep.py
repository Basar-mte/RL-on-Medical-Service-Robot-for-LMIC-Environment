"""Density generalization sweep (Table IV): evaluate every agent's best
checkpoint at low and high density (medium already in runs/results/table.json).
Agents were trained at medium only, so low/high measure generalization.
Writes runs/results/density_sweep.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.evaluate import evaluate_model

RUNS = ROOT / "runs"
AGENTS = ["cwn", "attention", "fixed_equal", "nav_heavy", "vanilla", "he2025"]
SEEDS = [0, 1, 2]
CONFIG = "configs/dmch_hospital.yaml"
N_EP = 100


def latest_run(agent, seed):
    ds = [d for d in RUNS.glob(f"{agent}_seed{seed}_*")
          if d.is_dir() and (d / "final_model.zip").exists()]
    return sorted(ds, key=lambda p: p.stat().st_mtime)[-1] if ds else None


def main():
    import numpy as np
    out = {}
    for density in ("low", "high"):
        out[density] = {}
        for agent in AGENTS:
            srs, cvrs = [], []
            for seed in SEEDS:
                d = latest_run(agent, seed)
                if d is None:
                    continue
                m = evaluate_model(str(d / "best_model"), CONFIG,
                                   n_episodes=N_EP, density=density)
                srs.append(m["sr"]); cvrs.append(m["cvr"])
            if srs:
                out[density][agent] = dict(
                    sr=float(np.mean(srs)), sr_std=float(np.std(srs)),
                    cvr=float(np.mean(cvrs)), cvr_std=float(np.std(cvrs)),
                    n_seeds=len(srs))
                print(f"[{density}] {agent}: SR={np.mean(srs):.3f} CVR={np.mean(cvrs):.4f}",
                      flush=True)
    (RUNS / "results").mkdir(parents=True, exist_ok=True)
    with open(RUNS / "results" / "density_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[density_sweep] wrote runs/results/density_sweep.json")


if __name__ == "__main__":
    main()
