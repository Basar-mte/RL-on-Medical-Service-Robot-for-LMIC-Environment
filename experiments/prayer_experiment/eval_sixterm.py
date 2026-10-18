"""Evaluate the 6-term weighter probe vs the 4-term CWN and fixed_equal on the
redesign env. Runs on the same env (patches via importing sixterm_train). Behaviour
is deterministic at eval, so the SR/CVR only reflect the policy learned under the
6-term reward. Verdict: does per-constraint weighting beat the 4-term / fixed?
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
sys.path.insert(0, str(ROOT))

import sixterm_train  # noqa: E402,F401  (redesign env + rich context + 6-term patches)
import envs.lmic_hospital_env as em  # noqa: E402
from eval.evaluate import _load_model_and_env, _batch, _unbatch, compute_spl  # noqa: E402

RUNS = ROOT / "sixterm_runs"
CONFIG = "prayer_experiment/dmch_redesign.yaml"
N_EP = 120
# redesign-campaign references (500K, 1 seed, same env)
REF = {"cwn_4term": (0.125, 0.0498), "attention": (0.133, 0.0574), "fixed_equal": (0.283, 0.0258)}


def eval_run(seed=0):
    ds = sorted([d for d in RUNS.glob(f"cwn_seed{seed}_*") if d.is_dir()], key=lambda p: p.stat().st_mtime)
    if not ds:
        return None
    d = ds[-1]
    md = d / "best_model" if (d / "best_model" / "best_model.zip").exists() else d
    model, inner, eval_env, _ = _load_model_and_env(str(md), CONFIG, agent="_probe")
    task_ids = em.TASK_IDS
    srs, cvrs, spls = [], [], []
    for ep in range(N_EP):
        task = task_ids[ep % len(task_ids)]
        eval_env.seed(6000 + ep); eval_env.set_options({"task": task, "density": "medium"})
        obs = _unbatch(eval_env.reset()); info = {}; n = 0
        while True:
            a, _ = model.predict(_batch(obs), deterministic=True)
            raw, _, dones, infos = eval_env.step(a[0][np.newaxis])
            obs = _unbatch(raw); info = infos[0]; n += 1
            if dones[0]:
                break
        succ = bool(info.get("task_complete", False))
        opt = info.get("path_length_optimal", 1.0); act = info.get("path_length_actual", opt)
        srs.append(1.0 if succ else 0.0); spls.append(compute_spl(succ, opt, act))
        cvrs.append(info.get("cultural_violations", 0) / max(n, 1))
    eval_env.close()
    return dict(sr=float(np.mean(srs)), cvr=float(np.mean(cvrs)), spl=float(np.mean(spls)), n=len(srs))


def main():
    print("=== 6-term weighter probe (redesign env, 120 ep) ===\n", flush=True)
    per = []
    for s in (0, 1, 2):
        r = eval_run(s)
        if r is not None:
            per.append(r); print(f"  seed{s}: SR={r['sr']:.3f} CVR={r['cvr']:.4f}", flush=True)
    if not per:
        print("no sixterm run found"); return
    m = dict(sr=float(np.mean([r["sr"] for r in per])), sr_std=float(np.std([r["sr"] for r in per])),
             cvr=float(np.mean([r["cvr"] for r in per])), spl=float(np.mean([r["spl"] for r in per])),
             n_seeds=len(per))
    print(f"\n6-term CWN : SR={m['sr']:.3f}+-{m['sr_std']:.3f}  CVR={m['cvr']:.4f}  ({m['n_seeds']} seeds)\n", flush=True)
    print("--- comparison (same env, 500K, seed 0) ---", flush=True)
    print(f"  6-term CWN     SR {m['sr']:.3f}  CVR {m['cvr']:.4f}", flush=True)
    for k, (sr, cvr) in REF.items():
        print(f"  {k:14s} SR {sr:.3f}  CVR {cvr:.4f}", flush=True)
    beats_4term = m["sr"] > REF["cwn_4term"][0] + 0.03
    beats_fixed = m["sr"] > REF["fixed_equal"][0] + 0.01
    print(f"\n  6-term beats 4-term CWN (>+3pt): {beats_4term} "
          f"({m['sr']:.3f} vs {REF['cwn_4term'][0]:.3f})", flush=True)
    print(f"  6-term beats fixed_equal        : {beats_fixed} "
          f"({m['sr']:.3f} vs {REF['fixed_equal'][0]:.3f})", flush=True)
    verdict = ("PROMISING - per-constraint weighting helps; extend to 3 seeds"
               if beats_fixed else
               "CLOSURE - finer weighting does NOT beat fixed weights; the ceiling is about the weighting AXIS, not its granularity")
    print(f"\n  VERDICT: {verdict}", flush=True)
    json.dump({"sixterm": m, "ref": REF, "beats_4term": bool(beats_4term),
               "beats_fixed": bool(beats_fixed), "verdict": verdict},
              open(RUNS / "sixterm_eval.json", "w"), indent=2)
    print(f"\nwrote {RUNS / 'sixterm_eval.json'}", flush=True)


if __name__ == "__main__":
    main()
