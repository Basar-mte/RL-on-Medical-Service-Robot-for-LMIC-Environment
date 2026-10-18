"""Evaluate the holonomic-ablation policies and recompute the adaptivity ceiling.

Runs under the SAME holonomic (unicycle) dynamics the policies trained on (the
monkeypatch is applied by importing holonomic_train), then reports, per agent,
SR/CVR/SPL and the per-task oracle-switcher ceiling, side by side with the
Ackermann results from runs/results/table.json. If CWN still does not beat the
attention weighter and the ceiling stays ~3 points, the Ackermann chassis is not
the reason reward-weighting fails to win.

Isolated: reads only holonomic_experiment/runs/ and runs/results/table.json;
writes only holonomic_experiment/runs/holonomic_eval.json.

Usage:  python holonomic_experiment/eval_holonomic.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import holonomic_experiment.holonomic_train  # noqa: E402,F401  (applies the kinematics patch on import)
from eval.evaluate import _load_model_and_env, _batch, _unbatch, compute_spl  # noqa: E402
from envs.lmic_hospital_env import TASK_IDS  # noqa: E402

RUNS = ROOT / "holonomic_experiment" / "runs"
CONFIG = "configs/dmch_hospital.yaml"
N_EP = 100
AGENTS = ["cwn", "attention", "vanilla"]


def find_run(agent, seed):
    ds = [d for d in RUNS.glob(f"{agent}_seed{seed}_*")
          if d.is_dir() and (d / "best_model" / "best_model.zip").exists()]
    return sorted(ds, key=lambda p: p.stat().st_mtime)[-1] if ds else None


def eval_one(agent, seed, density="medium"):
    d = find_run(agent, seed)
    if d is None:
        return None
    model, inner, eval_env, _ = _load_model_and_env(str(d / "best_model"), CONFIG)
    per_task = {t: [] for t in TASK_IDS}
    srs, cvrs, spls = [], [], []
    for ep in range(N_EP):
        task = TASK_IDS[ep % len(TASK_IDS)]
        eval_env.seed(1000 + ep)
        eval_env.set_options({"task": task, "density": density})
        obs = _unbatch(eval_env.reset())
        info = {}
        while True:
            a, _ = model.predict(_batch(obs), deterministic=True)
            raw, _, dones, infos = eval_env.step(a[0][np.newaxis])
            obs = _unbatch(raw); info = infos[0]
            if dones[0]:
                break
        steps = info.get("steps", 1)
        succ = bool(info.get("task_complete", False))
        opt = info.get("path_length_optimal", 1.0); act = info.get("path_length_actual", opt)
        per_task[task].append(succ)
        srs.append(1.0 if succ else 0.0)
        cvrs.append(info.get("cultural_violations", 0) / max(steps, 1))
        spls.append(compute_spl(succ, opt, act))
    eval_env.close()
    return dict(sr=float(np.mean(srs)), cvr=float(np.mean(cvrs)), spl=float(np.mean(spls)),
                per_task={t: (float(np.mean(v)) if v else 0.0) for t, v in per_task.items()})


def main():
    ack = json.load(open(ROOT / "runs" / "results" / "table.json"))
    out = {}
    task_sr = {a: {t: [] for t in TASK_IDS} for a in AGENTS}
    print("=== holonomic-kinematics ablation (medium density) ===", flush=True)
    for agent in AGENTS:
        srs, cvrs, spls, used = [], [], [], []
        for seed in (0, 1, 2):
            m = eval_one(agent, seed)
            if m is None:
                print(f"  [skip] {agent} seed{seed}: no best_model", flush=True)
                continue
            srs.append(m["sr"]); cvrs.append(m["cvr"]); spls.append(m["spl"]); used.append(seed)
            for t in TASK_IDS:
                task_sr[agent][t].append(m["per_task"][t])
            print(f"  {agent} seed{seed}: SR={m['sr']:.3f} CVR={m['cvr']:.4f}", flush=True)
        if not srs:
            continue
        out[agent] = dict(sr=float(np.mean(srs)), sr_std=float(np.std(srs)),
                          cvr=float(np.mean(cvrs)), spl=float(np.mean(spls)), n_seeds=len(used))
        a = ack.get(agent, {})
        print(f"  {agent.upper():9} Ackermann SR {a.get('sr',0)*100:5.1f}%  ->  "
              f"holonomic SR {out[agent]['sr']*100:5.1f}%  CVR {out[agent]['cvr']:.4f}", flush=True)

    # Per-task oracle ceiling (best agent per task type), holonomic
    present = [a for a in AGENTS if a in out]
    if present:
        per_task_mean = {a: {t: float(np.mean(task_sr[a][t])) for t in TASK_IDS} for a in present}
        oracle = float(np.mean([max(per_task_mean[a][t] for a in present) for t in TASK_IDS]))
        best_single = max(out[a]["sr"] for a in present)
        out["_ceiling"] = dict(per_task_oracle=oracle, best_single=best_single,
                               gain_pt=(oracle - best_single) * 100)
        print("\n--- adaptivity ceiling (holonomic) ---", flush=True)
        print(f"  best single agent SR : {best_single*100:.1f}%", flush=True)
        print(f"  per-task oracle SR   : {oracle*100:.1f}%", flush=True)
        print(f"  weighting gain       : {(oracle-best_single)*100:+.1f} pt   "
              f"(Ackermann was +3.3 pt: 50.7 -> 54.0)", flush=True)
        cwn = out.get("cwn", {}).get("sr", 0) * 100
        att = out.get("attention", {}).get("sr", 0) * 100
        print(f"  CWN vs attention     : {cwn:.1f}% vs {att:.1f}%  "
              f"({'CWN wins' if cwn > att else 'attention still >= CWN'})", flush=True)

    json.dump(out, open(RUNS / "holonomic_eval.json", "w"), indent=2)
    print("\nwrote holonomic_experiment/runs/holonomic_eval.json", flush=True)


if __name__ == "__main__":
    main()
