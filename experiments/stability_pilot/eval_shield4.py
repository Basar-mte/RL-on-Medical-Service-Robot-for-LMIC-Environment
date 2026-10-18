"""Inference-time safety shield on the pilot-4 FINAL models (stable recipe),
Tier-2 map, 500 episodes — same episode seeds / task cycling as
eval/eval_shielded.py — recording success / collision / timeout and CVR, so
the shielded numbers can be compared with the unshielded eval_final500.json.

    python experiments/stability_pilot/eval_shield4.py <agent> <seed>   # one run
    python experiments/stability_pilot/eval_shield4.py --summary
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
RUNS = HERE / "runs4"
CFG = "configs/hospital_v4.yaml"
N_EP = 500
AGENTS, SEEDS = ["fixed_equal", "vanilla", "attention"], [0, 1]


def run_dir(a, s):
    return sorted(p for p in (RUNS / f"{a}_s{s}").glob(f"{a}_seed{s}_*") if p.is_dir())[-1]


def evaluate(a, s):
    from eval.evaluate import _load_model_and_env, _batch, _unbatch
    from envs.lmic_hospital_env import TASK_IDS
    from baselines.safety_shield import shield_action
    rd = run_dir(a, s)
    model, inner, env, _ = _load_model_and_env(str(rd), CFG)   # run dir -> final_model.zip + vec_normalize.pkl
    recs = []
    for ep in range(N_EP):
        task = TASK_IDS[ep % len(TASK_IDS)]
        env.seed(1000 + ep)
        env.set_options({"task": task, "density": "medium"})
        obs = _unbatch(env.reset())
        while True:
            act, _ = model.predict(_batch(obs), deterministic=True)
            act = shield_action(act[0], {"lidar": inner._cast_lidar()})
            raw, _, dones, infos = env.step(act[np.newaxis])
            obs = _unbatch(raw); info = infos[0]
            if dones[0]:
                steps = max(info.get("steps", 1), 1)
                succ = bool(info.get("task_complete", False)); coll = bool(info.get("collision", False))
                recs.append(dict(task_id=task, success=succ, collision=coll,
                                 end="success" if succ else ("collision" if coll else "timeout"),
                                 cvr=info.get("cultural_violations", 0) / steps, steps=steps))
                break
    env.close()
    out = rd.parent / "eval_shield_final500.json"
    json.dump({"sr": float(np.mean([r["success"] for r in recs])), "records": recs}, open(out, "w"))
    print(f"{a}_s{s}: shielded SR {100*np.mean([r['success'] for r in recs]):.1f}%  "
          f"collision {100*np.mean([r['collision'] for r in recs]):.1f}%  -> {out.name}")


def summary():
    print(f"{'agent':<12}{'':<10}{'SR %':>8}{'collision %':>13}{'timeout %':>11}{'CVR':>8}{'STAT SR/CVR':>16}")
    for a in AGENTS:
        for tag, fname in (("no shield", "eval_final500.json"), ("+ shield", "eval_shield_final500.json")):
            recs = []
            for s in SEEDS:
                f = run_dir(a, s).parent / fname
                if f.exists():
                    recs += json.load(open(f))["records"]
            if not recs:
                continue
            stat = [r for r in recs if r["task_id"] == "STAT_medicine"]
            sr = 100 * np.mean([r["success"] for r in recs])
            coll = 100 * np.mean([r["end"] == "collision" for r in recs])
            tout = 100 * np.mean([r["end"] == "timeout" for r in recs])
            cvr = np.mean([r["cvr"] for r in recs])
            stat_s = f"{100*np.mean([r['success'] for r in stat]):.0f}% / {np.mean([r['cvr'] for r in stat]):.3f}"
            print(f"{a:<12}{tag:<10}{sr:>8.1f}{coll:>13.1f}{tout:>11.1f}{cvr:>8.3f}{stat_s:>16}")
    print("\npaper Tier-2 (old policies): no shield SR 24-28 / coll 72-76%;  Western shield SR 31.5-34.2 / coll 16-21%")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    else:
        evaluate(sys.argv[1], int(sys.argv[2]))
