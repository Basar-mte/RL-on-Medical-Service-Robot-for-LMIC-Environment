"""Evaluate the DMCH-redesign campaign + verify CWN is actually weighting.

Runs on the same redesign env + rich context + tasks the policies trained on
(patches applied by importing redesign_train). For every agent: SR / CVR / SPL
over all six tasks, with the prayer / gender / attendant CVR split. Then, for
CWN and attention, PROBES the trained weight-network with hand-crafted contexts
(prayer window on/off, inside vs far from a cultural zone) and reports whether
the output weights actually VARY with context (adaptive) or are ~constant
(saturated) — the "is CWN really weighting?" check.

Isolated: reads prayer_experiment/redesign_runs/ ; writes redesign_eval.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "prayer_experiment"))

import redesign_train  # noqa: E402,F401  (applies redesign env + rich context + task patches)
import envs.lmic_hospital_env as em  # noqa: E402
from eval.evaluate import _load_model_and_env, _batch, _unbatch, compute_spl  # noqa: E402

RUNS = ROOT / "prayer_experiment" / "redesign_runs"
CONFIG = "prayer_experiment/dmch_redesign.yaml"
AGENTS = ["cwn", "attention", "vanilla", "fixed_equal", "nav_heavy", "he2025"]
N_EP = 120
CULT = ("male_ward", "female_ward", "prayer")


def find_dir(agent):
    ds = sorted([d for d in RUNS.glob(f"{agent}_seed0_*") if d.is_dir()], key=lambda p: p.stat().st_mtime)
    if not ds:
        return None
    d = ds[-1]
    if (d / "best_model" / "best_model.zip").exists():
        return d / "best_model"
    return d if list(d.glob("*.zip")) else None


def eval_agent(agent):
    md = find_dir(agent)
    if md is None:
        print(f"  [skip] {agent}: no model", flush=True); return None
    model, inner, eval_env, _ = _load_model_and_env(str(md), CONFIG, agent="_probe")
    task_ids = em.TASK_IDS
    srs, cvrs, spls = [], [], []
    steps_tot = pr = ge = at = 0
    for ep in range(N_EP):
        task = task_ids[ep % len(task_ids)]
        eval_env.seed(5000 + ep)
        eval_env.set_options({"task": task, "density": "medium"})
        obs = _unbatch(eval_env.reset()); info = {}
        n = 0
        while True:
            a, _ = model.predict(_batch(obs), deterministic=True)
            raw, _, dones, infos = eval_env.step(a[0][np.newaxis])
            obs = _unbatch(raw); info = infos[0]
            zone = info.get("current_zone", "walkable"); cv = bool(info.get("cultural_violation", False))
            steps_tot += 1; n += 1
            if zone == "prayer": pr += 1
            elif zone in ("male_ward", "female_ward"): ge += 1
            elif cv: at += 1
            if dones[0]:
                break
        succ = bool(info.get("task_complete", False))
        opt = info.get("path_length_optimal", 1.0); act = info.get("path_length_actual", opt)
        srs.append(1.0 if succ else 0.0); spls.append(compute_spl(succ, opt, act))
        cvrs.append(info.get("cultural_violations", 0) / max(n, 1))
    eval_env.close()
    return dict(sr=float(np.mean(srs)), sr_std=float(np.std(srs)), cvr=float(np.mean(cvrs)),
                spl=float(np.mean(spls)), prayer_cvr=pr / max(steps_tot, 1),
                gender_cvr=ge / max(steps_tot, 1), attend_cvr=at / max(steps_tot, 1))


def probe_weighter(agent):
    """Load the trained weight net and check if weights vary with context."""
    md = find_dir(agent)
    if md is None:
        return None
    run = md.parent if md.name == "best_model" else md
    cands = ([md / f"{agent}_best.pt"] if (md / f"{agent}_best.pt").exists() else []) or \
        sorted(run.glob(f"{agent}_*.pt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return dict(note="no weight-net checkpoint")
    import torch
    from rewards.cwn import CWN
    from baselines.attention_weighting import AttentionWeightNet
    # build with context_dim=20 (rich context) before loading the 20-d checkpoint
    cls = CWN if agent == "cwn" else AttentionWeightNet
    net = cls(context_dim=20)
    data = torch.load(cands[-1], map_location="cpu", weights_only=False)
    net.load_state_dict(data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data)
    net.eval()
    # 20-d context probes: idx 13 cult_d, 14 prayer_active, 15 d_prayer, 17 in_cultural, 18 goal_in_cult
    def ctx(d):
        v = np.zeros(20, np.float32); v[1] = 0.6  # urgency
        for i, val in d.items():
            v[i] = val
        return v
    probes = {
        "neutral_far":       ctx({13: 1.0, 15: 1.0}),
        "prayer_on_near":    ctx({14: 1.0, 15: 0.0, 13: 0.0, 17: 1.0, 18: 1.0}),
        "prayer_off_near":   ctx({14: 0.0, 15: 0.0, 13: 0.0, 17: 1.0}),
        "in_gender_ward":    ctx({17: 1.0, 13: 0.0, 4: 1.0}),
        "crowded":           ctx({0: 1.0, 12: 1.0, 19: 1.0}),
    }
    weights = {k: np.asarray(net.get_weights(v), dtype=float).round(4).tolist() for k, v in probes.items()}
    W = np.array(list(weights.values()))
    per_dim_std = W.std(axis=0).round(4).tolist()
    temp = float(getattr(net, "temperature", getattr(net, "_temperature", float("nan"))))
    try:
        import torch
        if hasattr(net, "log_temperature"):
            temp = float(torch.exp(net.log_temperature).item())
    except Exception:
        pass
    return dict(temperature=temp, weights=weights, per_dim_std=per_dim_std,
                adaptive=bool(np.max(per_dim_std) > 0.02))


def main():
    out = {}
    print("=== DMCH-redesign evaluation (6 tasks, medium density) ===\n", flush=True)
    for a in AGENTS:
        print(f"[{a}] {N_EP} episodes ...", flush=True)
        m = eval_agent(a)
        if m is None:
            continue
        out[a] = m
        print(f"  SR={m['sr']:.3f}±{m['sr_std']:.3f}  CVR={m['cvr']:.4f}  SPL={m['spl']:.3f} | "
              f"prayer={m['prayer_cvr']:.4f} gender={m['gender_cvr']:.4f} attend={m['attend_cvr']:.4f}", flush=True)
    print("\n--- CWN / attention weighter probe (do weights vary with context?) ---", flush=True)
    for a in ("cwn", "attention"):
        try:
            p = probe_weighter(a)
        except Exception as e:
            p = dict(error=str(e))
        if p is None:
            continue
        out.setdefault(a, {})["weighter"] = p
        print(f"[{a}] temperature={p.get('temperature')}  adaptive={p.get('adaptive')}  "
              f"per-weight std={p.get('per_dim_std')}", flush=True)
        for k, w in p.get("weights", {}).items():
            print(f"    {k:16s} weights[nav,safe,cult,task] = {w}", flush=True)
    if "cwn" in out and "attention" in out:
        c, a = out["cwn"], out["attention"]
        print("\n--- summary ---", flush=True)
        print(f"  CWN   SR {c['sr']:.3f}  CVR {c['cvr']:.4f}  prayerCVR {c['prayer_cvr']:.4f}", flush=True)
        print(f"  attn  SR {a['sr']:.3f}  CVR {a['cvr']:.4f}  prayerCVR {a['prayer_cvr']:.4f}", flush=True)
        print(f"  CWN weighter adaptive: {out['cwn'].get('weighter', {}).get('adaptive')}", flush=True)
    json.dump(out, open(RUNS / "redesign_eval.json", "w"), indent=2)
    print(f"\nwrote {RUNS / 'redesign_eval.json'}", flush=True)


if __name__ == "__main__":
    main()
