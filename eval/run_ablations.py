"""
Phase 6.3 — Ablation runner.

Ablation study design (Table 3 in paper):
  A1 — Full CWN (baseline for ablation)
  A2 — CWN without prayer avoidance term (r_prayer = 0)
  A3 — CWN without gender-zone term (r_gender = 0)
  A4 — CWN without attendant proxemics term (r_attendant = 0)
  A5 — CWN with fixed equal weights (τ → ∞)
  A6 — No cultural reward at all (= B3 vanilla)

Also runs the density-level ablations (low / medium / high).

Usage:
  python eval/run_ablations.py --runs_dir runs/ --n_episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.lmic_hospital_env import LMICHospitalNavEnv, TASK_IDS, DENSITY_LEVELS
from rewards.cultural import prayer_avoidance_reward, gender_zone_reward, attendant_proxemics_reward
from eval.evaluate import evaluate_model, aggregate_metrics


ABLATION_REWARD_MODS = {
    "A1_full":         dict(no_prayer=False, no_gender=False, no_attendant=False),
    "A2_no_prayer":    dict(no_prayer=True,  no_gender=False, no_attendant=False),
    "A3_no_gender":    dict(no_prayer=False, no_gender=True,  no_attendant=False),
    "A4_no_attendant": dict(no_prayer=False, no_gender=False, no_attendant=True),
}

DENSITY_ABLATIONS = ["low", "medium", "high"]


def build_masked_cultural_reward(no_prayer: bool, no_gender: bool, no_attendant: bool):
    """Return a cultural_reward function with selected terms zeroed."""
    import numpy as _np

    def _masked_cultural(robot_pos, fp, tod_hour, nearest_family_dist, weights=None):
        if weights is None:
            weights = _np.array([1.0/3, 1.0/3, 1.0/3])

        r_prayer, v_prayer = (0.0, False) if no_prayer else prayer_avoidance_reward(robot_pos, fp, tod_hour)
        r_gender, v_gender = (0.0, False) if no_gender else gender_zone_reward(robot_pos, fp)
        r_att,    v_att    = (0.0, False) if no_attendant else attendant_proxemics_reward(nearest_family_dist)

        total = (weights[0] * r_prayer + weights[1] * r_gender + weights[2] * r_att)
        return float(_np.clip(total, -2.0, 0.0)), (v_prayer or v_gender or v_att)

    return _masked_cultural


def run_ablations(runs_dir: Path, n_episodes: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find CWN seed0 model — LATEST run directory by mtime (lexicographic
    # [0] picked the OLDEST run, i.e. a stale dir from a superseded campaign
    # whenever any older cwn_seed0_* remained in runs/), directories only
    # (the same glob also matches cwn_seed0_train.log launcher logs).
    cwn_dirs = sorted((p for p in runs_dir.glob("cwn_seed0_*") if p.is_dir()),
                      key=lambda p: p.stat().st_mtime)
    if not cwn_dirs:
        print("[ablations] No cwn_seed0_* run found — skipping model-based ablations.")
        return

    cwn_run = cwn_dirs[-1]
    model_dir = cwn_run / "best_model"
    model_path = model_dir / "best_model.zip"
    config = str(Path(__file__).parent.parent / "configs" / "dmch_hospital.yaml")

    if not model_path.exists():
        print(f"[ablations] Model not found at {model_path}. Skipping.")
        return

    results = {}

    # Density ablations
    print("\n[ablations] Density ablations...")
    for density in DENSITY_ABLATIONS:
        print(f"  density={density}")
        # evaluate_model(model_dir, config_path, ...) — takes the directory
        # containing the .zip (VecNormalize stats are auto-discovered via
        # _find_vecnormalize), not the .zip file path itself. Its return now
        # includes a "records" list (per-episode diagnostics) — serialized
        # as-is below on purpose.
        m = evaluate_model(str(model_dir), config, n_episodes=n_episodes, density=density)
        results[f"density_{density}"] = m
        with open(output_dir / f"ablation_density_{density}.json", "w") as f:
            json.dump({"summary": m}, f, indent=2)

    # Save combined
    with open(output_dir / "ablations_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ablations] Done. Results in {output_dir}")

    # Print summary
    print("\nAblation Results:")
    print(f"{'Config':<25} {'SR':>6} {'CVR':>7} {'SPL':>7}")
    for name, m in results.items():
        print(f"  {name:<23} {m['sr']:6.3f} {m['cvr']:7.3f} {m['spl']:7.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--output", default="results/ablations")
    args = p.parse_args()

    root = Path(__file__).parent.parent
    run_ablations(
        runs_dir=root / args.runs_dir,
        n_episodes=args.n_episodes,
        output_dir=root / args.output,
    )
