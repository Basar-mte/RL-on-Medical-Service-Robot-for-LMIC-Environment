"""
eval/log_trajectories.py — record one representative rollout per agent for
figures/make_figures.py's fig6_trajectories, replacing its previously
entirely-fabricated synthetic waypoints with real policy rollouts.

Does NOT touch the frozen 100-episode evaluation protocol in
eval/evaluate.py (aggregate_metrics()'s return contract is untouched) — this
is a separate, single-rollout-per-agent recording utility that reuses
eval.evaluate's Dict-obs-safe model/env loading.

Usage:
  python eval/log_trajectories.py --runs_dir runs/ \\
      --task_id STAT_medicine --density medium --seed 1000 \\
      --output_dir runs/results/trajectories
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluate import _load_model_and_env, _batch, _unbatch
from eval.post_training import AGENTS, find_latest_run


def record_trajectory(
    model_dir: str,
    config_path: str,
    task_id: str = "STAT_medicine",
    density: str = "medium",
    seed: int = 1000,
    deterministic: bool = True,
    max_steps: int = 1000,
) -> pd.DataFrame:
    """
    Run one deterministic rollout and return a DataFrame with columns
    [step, x, y, heading, collision, cultural_violation, social_violation, zone].

    Each row records the robot's pose BEFORE the action taken at that step,
    paired with the info dict describing the outcome of that action —
    i.e. row i is "from (x, y, heading), the policy acted, and this is what
    happened." The very last (post-terminal-action) position is not
    separately recorded: DummyVecEnv auto-resets on the terminal step, so by
    the time eval_env.step() returns, the wrapped LMICHospitalNavEnv already
    reflects the NEXT episode's spawn point — there is no way to read the
    true terminal (x, y) through the public VecEnv API once step() has
    returned. For a visualization spanning up to `max_steps` steps this is a
    negligible (one DT=0.1s step) truncation, not a scientific measurement.
    """
    model, inner_env, eval_env, _use_normalization = _load_model_and_env(model_dir, config_path)

    eval_env.seed(seed)
    eval_env.set_options({"task": task_id, "density": density})
    obs = _unbatch(eval_env.reset())

    rows = []
    done = False
    step = 0
    while not done and step < max_steps:
        x, y = float(inner_env._robot_pos[0]), float(inner_env._robot_pos[1])
        heading = float(inner_env._robot_heading)

        action, _ = model.predict(_batch(obs), deterministic=deterministic)
        action = action[0]
        raw_obs, _, dones, infos = eval_env.step(action[np.newaxis])
        info = infos[0] if infos else {}

        rows.append(dict(
            step=step, x=x, y=y, heading=heading,
            collision=bool(info.get("collision", False)),
            cultural_violation=bool(info.get("cultural_violation", False)),
            social_violation=bool(info.get("social_violation", False)),
            zone=info.get("current_zone"),
        ))

        obs = _unbatch(raw_obs)
        done = bool(dones[0])
        step += 1

    eval_env.close()
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--config", default="configs/dmch_hospital.yaml")
    p.add_argument("--task_id", default="STAT_medicine")
    p.add_argument("--density", default="medium")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output_dir", default="runs/results/trajectories")
    args = p.parse_args()

    root = Path(__file__).parent.parent
    runs_dir = root / args.runs_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Dict] = {}
    for agent in AGENTS:
        # seed=0 for consistency with run_ablations.py's / post_training.py's
        # "seed0 is the representative run" convention.
        run_dir = find_latest_run(runs_dir, agent, seed=0)
        if run_dir is None:
            print(f"  [skip] no {agent}_seed0_* run found under {runs_dir}")
            continue

        model_dir = run_dir / "best_model"
        if not (model_dir / "best_model.zip").exists():
            if (run_dir / "final_model.zip").exists():
                model_dir = run_dir
            else:
                print(f"  [skip] no best_model/final_model in {run_dir.name}")
                continue

        print(f"  [{agent}] recording rollout from {run_dir.name}...")
        df = record_trajectory(
            str(model_dir), args.config,
            task_id=args.task_id, density=args.density, seed=args.seed,
        )
        csv_path = output_dir / f"{agent}_traj.csv"
        df.to_csv(csv_path, index=False)

        n_viol = int((df["cultural_violation"] | df["social_violation"]).sum())
        ended_in_collision = bool(len(df) > 0 and df["collision"].iloc[-1])
        meta[agent] = dict(
            run_dir=run_dir.name, model_dir=str(model_dir.relative_to(root)),
            task_id=args.task_id, density=args.density, seed=args.seed,
            n_steps=len(df), n_violations=n_viol,
            ended_in_collision=ended_in_collision,
        )
        print(f"    -> {csv_path.relative_to(root)} ({len(df)} steps, {n_viol} violations)")

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[log_trajectories] Done. {len(meta)}/{len(AGENTS)} agents recorded. "
          f"Meta saved to {output_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
