"""Evaluate agents with the reactive safety shield applied at inference.

Reuses existing trained checkpoints (no retraining). For each requested
agent, runs 100 episodes x 3 seeds at medium density with the shield
wrapping the policy action, and prints SR/CVR vs the unshielded baseline
from runs/results/table.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.evaluate import _load_model_and_env, _batch, _unbatch, aggregate_metrics, compute_spl
from envs.lmic_hospital_env import TASK_IDS
from baselines.safety_shield import shield_action

RUNS = ROOT / "runs"
CONFIG = "configs/dmch_hospital.yaml"
N_EP = 100


def latest_run(agent, seed):
    ds = [d for d in RUNS.glob(f"{agent}_seed{seed}_*")
          if d.is_dir() and (d / "final_model.zip").exists()]
    return sorted(ds, key=lambda p: p.stat().st_mtime)[-1] if ds else None


def eval_one(agent, seed, density="medium"):
    d = latest_run(agent, seed)
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
            # The shield needs RAW LiDAR in metres. obs["lidar"] here is
            # VecNormalize-whitened (what the policy consumes), so read the
            # unnormalised [0,1]-over-range scan straight from the inner env
            # at the current state instead.
            raw_obs = {"lidar": inner._cast_lidar()}
            a = shield_action(a[0], raw_obs)
            raw, _, dones, infos = eval_env.step(a[np.newaxis])
            obs = _unbatch(raw); info = infos[0]
            if dones[0]:
                break
        steps = info.get("steps", 1)
        succ = bool(info.get("task_complete", False))
        opt = info.get("path_length_optimal", 1.0); act = info.get("path_length_actual", opt)
        records.append(dict(success=succ, cvr=info.get("cultural_violations", 0) / max(steps, 1),
                            svr=info.get("social_violations", 0) / max(steps, 1),
                            spl=compute_spl(succ, opt, act), mpd=info.get("min_pedestrian_distance", 0.0),
                            ct_s=steps * 0.1))
    eval_env.close()
    return aggregate_metrics(records)


def main():
    agents = sys.argv[1:] or ["attention", "vanilla"]
    base = json.load(open(RUNS / "results" / "table.json"))
    out = {}
    for agent in agents:
        srs, cvrs, spls = [], [], []
        for seed in (0, 1, 2):
            m = eval_one(agent, seed)
            srs.append(m["sr"]); cvrs.append(m["cvr"]); spls.append(m["spl"])
            print(f"  {agent} seed{seed}: SR={m['sr']:.3f} CVR={m['cvr']:.4f}", flush=True)
        sr, cvr, spl = np.mean(srs), np.mean(cvrs), np.mean(spls)
        b = base.get(agent, {})
        out[agent + "+shield"] = dict(sr=float(sr), sr_std=float(np.std(srs)),
                                      cvr=float(cvr), cvr_std=float(np.std(cvrs)),
                                      spl=float(spl), n_seeds=len(srs))
        print(f"{agent}+SHIELD: SR {b.get('sr',0):.3f}->{sr:.3f}  "
              f"CVR {b.get('cvr',0):.4f}->{cvr:.4f}  SPL {b.get('spl',0):.3f}->{spl:.3f}", flush=True)
    (RUNS / "results").mkdir(parents=True, exist_ok=True)
    with open(RUNS / "results" / "shielded.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote runs/results/shielded.json")


if __name__ == "__main__":
    main()
