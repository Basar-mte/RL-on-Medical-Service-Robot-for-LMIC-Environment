"""Evaluate the shield-in-loop (BONUS) policies and compare to the inference-
time shield baseline.

Isolation: reads only shield_experiment/runs/ (policies trained WITH the shield
wrapped into the env) and runs/results/shielded.json (the inference-shield
baseline); writes only shield_experiment/runs/shield_loop_eval.json. Does not
touch the main pipeline, the paper, or runs/.

Protocol matches eval/eval_shielded.py exactly (100 episodes, tasks cycled,
shared eval seeds 1000+ep, deterministic actions, shield applied to raw LiDAR),
so the shield-in-loop numbers are directly comparable to the inference-shield
numbers in shielded.json. The only difference is how the policy was trained.

Usage:  python shield_experiment/eval_shield_loop.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.evaluate import (_load_model_and_env, _batch, _unbatch,          # noqa: E402
                           aggregate_metrics, compute_spl)
from envs.lmic_hospital_env import TASK_IDS                                # noqa: E402
from baselines.safety_shield import shield_action                          # noqa: E402

RUNS = ROOT / "shield_experiment" / "runs"
CONFIG = "configs/dmch_hospital.yaml"
N_EP = 100
AGENTS = ["attention", "cwn", "vanilla"]


def find_run(agent, seed):
    """Newest shield_experiment run for (agent, seed) that has a best_model
    (final_model not required -- eval uses the best validation checkpoint)."""
    ds = [d for d in RUNS.glob(f"{agent}_seed{seed}_*")
          if d.is_dir() and (d / "best_model" / "best_model.zip").exists()]
    return sorted(ds, key=lambda p: p.stat().st_mtime)[-1] if ds else None


def eval_one(agent, seed, density="medium"):
    d = find_run(agent, seed)
    if d is None:
        return None
    model, inner, eval_env, _ = _load_model_and_env(str(d / "best_model"), CONFIG)
    records = []
    for ep in range(N_EP):
        task = TASK_IDS[ep % len(TASK_IDS)]
        eval_env.seed(1000 + ep)
        eval_env.set_options({"task": task, "density": density})
        obs = _unbatch(eval_env.reset())
        info = {}
        while True:
            a, _ = model.predict(_batch(obs), deterministic=True)
            a = shield_action(a[0], {"lidar": inner._cast_lidar()})  # raw LiDAR
            raw, _, dones, infos = eval_env.step(a[np.newaxis])
            obs = _unbatch(raw); info = infos[0]
            if dones[0]:
                break
        steps = info.get("steps", 1)
        succ = bool(info.get("task_complete", False))
        opt = info.get("path_length_optimal", 1.0); act = info.get("path_length_actual", opt)
        records.append(dict(success=succ,
                            cvr=info.get("cultural_violations", 0) / max(steps, 1),
                            svr=info.get("social_violations", 0) / max(steps, 1),
                            spl=compute_spl(succ, opt, act),
                            mpd=info.get("min_pedestrian_distance", 0.0),
                            ct_s=steps * 0.1))
    eval_env.close()
    return aggregate_metrics(records)


def main():
    base = json.load(open(ROOT / "runs" / "results" / "shielded.json"))
    out = {}
    print("=== shield-in-loop eval (bonus) vs inference-shield baseline ===", flush=True)
    for agent in AGENTS:
        srs, cvrs, spls, used = [], [], [], []
        for seed in (0, 1, 2):
            m = eval_one(agent, seed)
            if m is None:
                print(f"  [skip] {agent} seed{seed}: no best_model", flush=True)
                continue
            srs.append(m["sr"]); cvrs.append(m["cvr"]); spls.append(m["spl"]); used.append(seed)
            print(f"  {agent} seed{seed}: SR={m['sr']:.3f} CVR={m['cvr']:.4f}", flush=True)
        if not srs:
            continue
        sr, cvr, spl = float(np.mean(srs)), float(np.mean(cvrs)), float(np.mean(spls))
        b = base.get(agent + "+shield", {})
        out[agent + "+shield_loop"] = dict(
            sr=sr, sr_std=float(np.std(srs)), cvr=cvr, cvr_std=float(np.std(cvrs)),
            spl=spl, n_seeds=len(srs), seeds=used)
        print(f"  {agent.upper():10} inference-shield SR {b.get('sr',0)*100:5.1f}% "
              f"CVR {b.get('cvr',0):.4f}  ->  shield-in-loop SR {sr*100:5.1f}% CVR {cvr:.4f}  "
              f"(dSR {(sr-b.get('sr',0))*100:+.1f} pt)", flush=True)
    json.dump(out, open(RUNS / "shield_loop_eval.json", "w"), indent=2)
    print("wrote shield_experiment/runs/shield_loop_eval.json", flush=True)


if __name__ == "__main__":
    main()
