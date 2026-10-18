"""Evaluate the prayer-ENGAGEMENT probe and answer the one question that matters:

  Does a LEARNED weighter (CWN) reduce prayer-zone crossings DURING prayer
  windows more than fixed weights can — i.e. does it exploit the time-varying
  prayer context that fixed weights structurally cannot see?

Runs on the SAME engaged benchmark the policies trained on (patches applied by
importing prayer_engage). Behaviour at eval is fully determined by the SAC policy
(deterministic); reward weights do not affect the path, so the weight-net is not
loaded (agent="_probe" => weight_source=None) — this measures where each policy
actually goes.

For the engaging task (prayer_call: short route = west door THROUGH the prayer
gate; long route = south door AROUND it), every step is conditioned on whether
its time-of-day falls in a prayer window:

  prayer-CVR | in-window   vs   prayer-CVR | out-window
  adaptivity  = prayer-CVR(out) - prayer-CVR(in)   (>0 => avoids more in-window)

A context-blind fixed weighter should show adaptivity ~ 0; a working learned
weighter should show a clear positive gap. Gender-CVR is a control (both routes
reach the female ward). SR guards against "avoid prayer by failing the task".

Isolated: reads only prayer_experiment/runs/ + the env; writes
prayer_experiment/runs/prayer_eval.json.

Usage:  .venv\\Scripts\\python.exe prayer_experiment\\eval_prayer.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "prayer_experiment"))

import prayer_engage as pe   # noqa: E402  (applies engaged-env + richer-context patches)
from eval.evaluate import _load_model_and_env, _batch, _unbatch, compute_spl  # noqa: E402

RUNS = ROOT / "prayer_experiment" / "runs"
CONFIG = "configs/dmch_hospital.yaml"
AGENTS = ["cwn", "attention", "fixed_equal"]
N_EP = 300                      # prayer_call episodes per agent
CULT = {"prayer", "male_ward", "female_ward"}


def find_model_dir(agent):
    ds = sorted([d for d in RUNS.glob(f"{agent}_seed0_*") if d.is_dir()],
                key=lambda p: p.stat().st_mtime)
    if not ds:
        return None
    d = ds[-1]
    if (d / "best_model" / "best_model.zip").exists():
        return d / "best_model"
    if (d / "final_model.zip").exists() or list(d.glob("*.zip")):
        return d
    return None


def eval_agent(agent):
    md = find_model_dir(agent)
    if md is None:
        print(f"  [skip] {agent}: no model", flush=True)
        return None
    model, inner, eval_env, _ = _load_model_and_env(str(md), CONFIG, agent="_probe")
    fp = inner.fp

    # step-conditioned accumulators
    acc = {w: dict(steps=0, prayer=0, gender=0, attend=0) for w in ("in", "out")}
    # episode-level route + task
    ep_stats = []   # (in_window_majority, took_west, success)
    srs, spls, cvrs = [], [], []

    for ep in range(N_EP):
        eval_env.seed(4000 + ep)
        eval_env.set_options({"task": pe.PRAYER_TASK, "density": "medium"})
        obs = _unbatch(eval_env.reset())
        info = {}
        ep_in = ep_out = ep_prayer = 0
        steps = 0
        while True:
            a, _ = model.predict(_batch(obs), deterministic=True)
            raw, _, dones, infos = eval_env.step(a[0][np.newaxis])
            obs = _unbatch(raw); info = infos[0]
            zone = info.get("current_zone", "walkable")
            tod = float(info.get("tod_hour", 12.0))
            in_win = fp.is_prayer_window(tod)
            cv = bool(info.get("cultural_violation", False))
            w = "in" if in_win else "out"
            acc[w]["steps"] += 1
            if zone == "prayer":
                acc[w]["prayer"] += 1; ep_prayer += 1
            elif zone in ("male_ward", "female_ward"):
                acc[w]["gender"] += 1
            elif cv:
                acc[w]["attend"] += 1
            if in_win: ep_in += 1
            else: ep_out += 1
            steps += 1
            if dones[0]:
                break
        succ = bool(info.get("task_complete", False))
        opt = info.get("path_length_optimal", 1.0); act = info.get("path_length_actual", opt)
        srs.append(1.0 if succ else 0.0)
        spls.append(compute_spl(succ, opt, act))
        cvrs.append(info.get("cultural_violations", 0) / max(steps, 1))
        ep_stats.append((ep_in >= ep_out, ep_prayer > 0, succ))
    eval_env.close()

    def cvr(w, kind):
        s = acc[w]["steps"]
        return acc[w][kind] / s if s else 0.0

    pin, pout = cvr("in", "prayer"), cvr("out", "prayer")
    ep = np.array(ep_stats, dtype=object)
    inm = np.array([e[0] for e in ep_stats]); west = np.array([e[1] for e in ep_stats])
    west_in = float(west[inm].mean()) if inm.any() else 0.0
    west_out = float(west[~inm].mean()) if (~inm).any() else 0.0

    return dict(
        sr=float(np.mean(srs)), spl=float(np.mean(spls)), cvr=float(np.mean(cvrs)),
        prayer_cvr_in=pin, prayer_cvr_out=pout, prayer_adaptivity=pout - pin,
        gender_cvr_in=cvr("in", "gender"), gender_cvr_out=cvr("out", "gender"),
        attend_cvr_in=cvr("in", "attend"), attend_cvr_out=cvr("out", "attend"),
        west_door_in=west_in, west_door_out=west_out,
        n_in_steps=acc["in"]["steps"], n_out_steps=acc["out"]["steps"],
        n_prayer_episodes=int(inm.sum()),
    )


def main():
    out = {}
    print("=== prayer-engagement probe — prayer_call task, medium density ===\n", flush=True)
    for a in AGENTS:
        print(f"[{a}] evaluating {N_EP} episodes ...", flush=True)
        m = eval_agent(a)
        if m is None:
            continue
        out[a] = m
        print(f"  SR={m['sr']:.3f}  SPL={m['spl']:.3f}  CVR={m['cvr']:.4f}", flush=True)
        print(f"  prayer-CVR  in-window={m['prayer_cvr_in']:.4f}  "
              f"out-window={m['prayer_cvr_out']:.4f}  "
              f"ADAPTIVITY={m['prayer_adaptivity']:+.4f}", flush=True)
        print(f"  west-door use  in-window={m['west_door_in']:.2f}  "
              f"out-window={m['west_door_out']:.2f}   "
              f"(prayer episodes n={m['n_prayer_episodes']})", flush=True)
        print(f"  gender-CVR  in={m['gender_cvr_in']:.3f} out={m['gender_cvr_out']:.3f}\n", flush=True)

    if {"cwn", "attention", "fixed_equal"} <= set(out):
        print("--- VERDICT ------------------------------------------------------", flush=True)
        c, at, fx = out["cwn"], out["attention"], out["fixed_equal"]
        print(f"  adaptivity (prayer-CVR drop in prayer windows):", flush=True)
        print(f"    CWN         {c['prayer_adaptivity']:+.4f}", flush=True)
        print(f"    attention   {at['prayer_adaptivity']:+.4f}", flush=True)
        print(f"    fixed_equal {fx['prayer_adaptivity']:+.4f}  (context-blind baseline)", flush=True)
        cwn_wins_adapt = (c["prayer_adaptivity"] > at["prayer_adaptivity"]
                          and c["prayer_adaptivity"] > fx["prayer_adaptivity"] + 0.005)
        cwn_cvr_ok = c["cvr"] <= at["cvr"] + 1e-6
        print(f"\n  CWN shows the largest prayer-window adaptivity : {cwn_wins_adapt}", flush=True)
        print(f"  CWN overall CVR <= attention                   : {cwn_cvr_ok} "
              f"(CWN {c['cvr']:.4f} vs attn {at['cvr']:.4f})", flush=True)
        print(f"  CWN SR within 3 pt of best                     : "
              f"{c['sr'] >= max(at['sr'], fx['sr']) - 0.03} "
              f"(CWN {c['sr']:.3f}, best {max(at['sr'],fx['sr']):.3f})", flush=True)
        verdict = "PROMISING -> worth the full re-run" if (cwn_wins_adapt and cwn_cvr_ok) \
            else "FLAT -> mechanism does not lift CWN; keep the honest note"
        print(f"\n  PROBE VERDICT: {verdict}", flush=True)
        out["_verdict"] = dict(cwn_wins_adaptivity=bool(cwn_wins_adapt),
                               cwn_cvr_ok=bool(cwn_cvr_ok), text=verdict)

    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RUNS / "prayer_eval.json", "w"), indent=2)
    print(f"\nwrote {RUNS / 'prayer_eval.json'}", flush=True)


if __name__ == "__main__":
    main()
