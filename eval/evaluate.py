"""
Phase 6.1 — Frozen evaluation protocol.

Evaluates a saved SAC model on LMICHospitalNav-v1 across all task types
and density levels. Computes:
  • SR   — Success Rate
  • CVR  — Cultural Violation Rate (novel metric, C2)
  • SPL  — Success-weighted Path Length (Anderson 2018)
  • MPD  — Mean Minimum Pedestrian Distance
  • SVR  — Social Violation Rate
  • CT   — Completion Time (seconds)

Usage:
  python eval/evaluate.py --model_dir runs/cwn_seed0_XXX/best_model \\
         --n_episodes 100 --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.lmic_hospital_env import LMICHospitalNavEnv, TASK_IDS, DENSITY_LEVELS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_spl(success: bool, optimal_len: float, actual_len: float) -> float:
    if not success or actual_len < 1e-6:
        return 0.0
    return optimal_len / max(actual_len, optimal_len)


def aggregate_metrics(records: List[Dict]) -> Dict:
    sr   = np.mean([r["success"] for r in records])
    cvr  = np.mean([r["cvr"] for r in records])
    spl  = np.mean([r["spl"] for r in records])
    mpd  = np.mean([r["mpd"] for r in records])
    svr  = np.mean([r["svr"] for r in records])
    ct_s = np.mean([r["ct_s"] for r in records if r["success"]] or [0])
    return dict(
        sr=float(sr),
        cvr=float(cvr),
        spl=float(spl),
        mpd=float(mpd),
        svr=float(svr),
        ct_s=float(ct_s),
        n_episodes=len(records),
    )


# ---------------------------------------------------------------------------
# VecNormalize helper
# ---------------------------------------------------------------------------

def _find_vecnormalize(model_dir: Path) -> Optional[Path]:
    """
    Search for VecNormalize stats alongside the saved model.

    Search order:
      1. run_dir / "vec_normalize.pkl"          (written at end of training)
      2. run_dir / "checkpoints/*_vecnormalize.pkl"  (latest checkpoint)
    where run_dir is model_dir or its parent.
    """
    candidates = []
    for base in [model_dir, model_dir.parent]:
        vn = base / "vec_normalize.pkl"
        if vn.exists():
            candidates.append(vn)
        for pkl in sorted(base.glob("checkpoints/*_vecnormalize.pkl")):
            candidates.append(pkl)

    if not candidates:
        return None
    # Return the most recently modified one
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    model_dir: str,
    config_path: str,
    n_episodes: int = 100,
    density: str = "medium",
    deterministic: bool = True,
) -> Dict:
    """
    Run `n_episodes` rollouts with the saved model and return metrics dict.
    """
    model_dir = Path(model_dir)

    # Find the .zip model file
    zip_files = list(model_dir.glob("*.zip"))
    if not zip_files:
        # EvalCallback saves best_model.zip directly inside the dir
        zip_files = list(model_dir.parent.glob("best_model.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model found in {model_dir}")
    model_path = sorted(zip_files, key=lambda p: p.stat().st_mtime)[-1]

    # Find VecNormalize
    vn_path = _find_vecnormalize(model_dir)
    use_normalization = vn_path is not None
    if use_normalization:
        print(f"  [eval] Using VecNormalize from: {vn_path.name}")
    else:
        print("  [eval] No VecNormalize found — evaluating without obs normalization")

    # Load model (on CPU for evaluation)
    model = SAC.load(str(model_path), device="cpu")

    # Build environment
    cfg = str(Path(__file__).parent.parent / config_path)
    env_fn = lambda: Monitor(LMICHospitalNavEnv(floorplan_path=cfg))

    if use_normalization:
        raw_env = DummyVecEnv([env_fn])
        eval_env = VecNormalize.load(str(vn_path), raw_env)
        eval_env.training = False
        eval_env.norm_reward = False
        inner_env = eval_env.envs[0].env  # unwrap Monitor → LMICHospitalNavEnv
    else:
        inner_env = LMICHospitalNavEnv(floorplan_path=cfg)
        eval_env = None

    records = []
    for ep in range(n_episodes):
        task_id = TASK_IDS[ep % len(TASK_IDS)]

        if use_normalization:
            # Reset the inner env with task options
            raw_obs, _ = inner_env.reset(seed=1000 + ep,
                                          options={"task": task_id, "density": density})
            # Sync the VecNormalize wrapper so its internal state is correct
            eval_env.envs[0].reset()
            obs = eval_env.normalize_obs(raw_obs[np.newaxis])[0]
        else:
            obs, _ = inner_env.reset(seed=1000 + ep,
                                      options={"task": task_id, "density": density})

        terminated = truncated = False
        info_last: Dict = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs[np.newaxis], deterministic=deterministic)
            action = action[0]

            if use_normalization:
                raw_obs, _, terminated_arr, truncated_arr, infos = eval_env.step(
                    action[np.newaxis]
                )
                obs = raw_obs[0]
                terminated = bool(terminated_arr[0])
                truncated = bool(truncated_arr[0])
                info_last = infos[0] if infos else {}
            else:
                obs, _, terminated, truncated, info_last = inner_env.step(action)

        steps    = info_last.get("steps", 1)
        success  = bool(info_last.get("task_complete", False))
        cult_viol = info_last.get("cultural_violations", 0)
        soc_viol  = info_last.get("social_violations", 0)
        opt_len   = info_last.get("path_length_optimal", 1.0)
        act_len   = info_last.get("path_length_actual", opt_len)
        mpd_val   = info_last.get("min_pedestrian_distance", 0.0)

        records.append(dict(
            ep=ep,
            task_id=task_id,
            success=success,
            cvr=cult_viol / max(steps, 1),
            svr=soc_viol  / max(steps, 1),
            spl=compute_spl(success, opt_len, act_len),
            mpd=float(mpd_val),
            ct_s=steps * 0.1,
        ))

        if (ep + 1) % 10 == 0:
            partial = aggregate_metrics(records)
            print(f"  ep {ep+1:3d}/{n_episodes}  SR={partial['sr']:.3f}  "
                  f"CVR={partial['cvr']:.4f}  SPL={partial['spl']:.3f}  "
                  f"MPD={partial['mpd']:.3f}")

    if use_normalization:
        eval_env.close()
    else:
        inner_env.close()

    return aggregate_metrics(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="Path to model directory (containing .zip)")
    p.add_argument("--config", default="configs/dmch_hospital.yaml")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--density", default="medium")
    p.add_argument("--output", default="eval_results.json",
                   help="Output JSON filename (saved next to model_dir)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_model(
        model_dir=args.model_dir,
        config_path=args.config,
        n_episodes=args.n_episodes,
        density=args.density,
    )
    out_path = Path(args.model_dir).parent / args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate] Saved to {out_path}")
    print(json.dumps(results, indent=2))
