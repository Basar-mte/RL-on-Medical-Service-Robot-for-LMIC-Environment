"""Evaluate one group of the final campaign (runs_v5/<group>/): paper protocol
with more episodes and every checkpoint that matters:
  final      FINAL model (2M for main tiers, 1M for the pivot ablation)   500 eps
  mid        the 1M checkpoint of 2M runs (budget-matched to the paper)  500 eps
  best       best-by-eval-reward checkpoint (reference only)             500 eps
  shield     FINAL model + inference safety shield                        500 eps
Deterministic, episode seeds 1000+ep, tasks cycled. Pivot-group runs are
evaluated under the same pivot kinematics they were trained with (PILOT_PIVOT=1
is inherited from the campaign; evaluate.py runs through patch_sac.py).
Aggregates to runs_v5/results/<group>/{table,per_task,per_seed}.json + EVAL_DONE.
Idempotent: existing per-run eval files are reused. Per-episode records are
kept so further metrics can be computed later without re-running anything.

    python scripts/eval_v5.py tier2          # (or tier1, tier2_pivot)
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PATCH = str(ROOT / "experiments" / "stability_pilot" / "patch_sac.py")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import campaign_v5_stable as C  # noqa: E402  (GROUPS / RUNS / SMOKE are the single source of truth)

N_EP = 3 if C.SMOKE else 500
PARALLEL = C.EVAL_PROCS   # the campaign lowers its training cap by this while evaluating (CPU <= ~90%)


def run_dir(g, a, s):
    return C.run_dir((g, a, s))


def prepare_mid(g, rd):
    """Copy the mid (1M) checkpoint into <run>/ckpt_mid/ as a loadable model dir."""
    st = C.GROUPS[g].get("mid")
    if st is None:
        return None
    dst = rd / "ckpt_mid"
    prefix = "ppo" if C.GROUPS[g].get("trainer") == C.PPO else "sac"
    z, v = rd / "checkpoints" / f"{prefix}_{st}_steps.zip", rd / "checkpoints" / f"{prefix}_vecnormalize_{st}_steps.pkl"
    if not (z.exists() and v.exists()):
        return None
    dst.mkdir(exist_ok=True)
    if not (dst / "model.zip").exists():
        shutil.copy(z, dst / "model.zip")
        shutil.copy(v, dst / "vec_normalize.pkl")
        for pt in rd.glob(f"*_{st}.pt"):           # adaptive agents' weight net at that step
            shutil.copy(pt, dst / pt.name)
    return dst


def shield_eval(g, a, s, v2=False):
    """v2=False: baselines/safety_shield.py (forward-cone CBF, the paper's original shield);
    v2=True: experiments/planner/arc_shield.py (kinematic swept-path shield, Oct 2026)."""
    from eval.evaluate import _load_model_and_env, _batch, _unbatch, compute_spl
    from envs.lmic_hospital_env import TASK_IDS
    from baselines.safety_shield import shield_action
    sys.path.insert(0, str(ROOT / "experiments"))
    from planner.arc_shield import shield_action as shield_v2
    rd = run_dir(g, a, s)
    model, inner, env, _ = _load_model_and_env(str(rd), C.GROUPS[g]["config"])
    # Residual agents: the policy output is a correction on the social-force prior, so the
    # shield is applied inside the env to the executed action (planner/residual.py).
    residual = bool(getattr(type(inner), "_residual_patched", False))
    if residual:
        inner._env_shield = "v2" if v2 else "v1"
    recs = []
    for ep in range(N_EP):
        task = TASK_IDS[ep % len(TASK_IDS)]
        env.seed(1000 + ep); env.set_options({"task": task, "density": "medium"})
        obs = _unbatch(env.reset())
        while True:
            act, _ = model.predict(_batch(obs), deterministic=True)
            if residual:
                act = act[0]
            elif v2:
                act = shield_v2(act[0], {"lidar": inner._cast_lidar(), "pedestrians": inner._ped_obs_vec(),
                                         "robot_state": inner._robot_state_vec()})
            else:
                act = shield_action(act[0], {"lidar": inner._cast_lidar()})
            raw, _, dones, infos = env.step(act[np.newaxis]); obs = _unbatch(raw); info = infos[0]
            if dones[0]:
                steps = max(info.get("steps", 1), 1)
                succ = bool(info.get("task_complete")); coll = bool(info.get("collision"))
                opt = info.get("path_length_optimal", 1.0); actual = info.get("path_length_actual", opt)
                recs.append(dict(task_id=task, success=succ, end="success" if succ else ("collision" if coll else "timeout"),
                                 cvr=info.get("cultural_violations", 0) / steps, svr=info.get("social_violations", 0) / steps,
                                 spl=compute_spl(succ, opt, actual), mpd=info.get("min_pedestrian_distance", 0.0),
                                 steps=steps))
                break
    env.close()
    json.dump({"sr": float(np.mean([r["success"] for r in recs])), "records": recs},
              open(rd.parent / ("eval_shield2.json" if v2 else "eval_shield.json"), "w"))


def jobs(g):
    spec = C.GROUPS[g]
    for a in spec["agents"]:
        for s in spec["seeds"]:
            rd = run_dir(g, a, s)
            if rd is None or not (rd / "final_model.zip").exists():
                raise SystemExit(f"{g}/{a}_s{s} not complete")
            key = (g, a, s)
            ev = lambda md, out: [PATCH, "eval/evaluate.py", "--model_dir", str(md), "--config", spec["config"],  # noqa: E731
                                  "--n_episodes", str(N_EP), "--output", out]
            # evaluate.py writes to <model_dir>.parent/<output>
            if not (rd.parent / "eval_final.json").exists():
                yield ev(rd, "eval_final.json"), rd.parent / "eval_final.log", key
            mid = prepare_mid(g, rd)
            if mid is not None and not (rd / "eval_mid.json").exists():
                yield ev(mid, "eval_mid.json"), rd.parent / "eval_mid.log", key
            if (rd / "best_model").exists() and not (rd / "eval_best.json").exists():
                yield ev(rd / "best_model", "eval_best.json"), rd.parent / "eval_best.log", key
            if not (rd.parent / "eval_shield.json").exists():
                yield [PATCH, "scripts/eval_v5.py", "--shield", g, a, str(s)], rd.parent / "eval_shield.log", key
            if not (rd.parent / "eval_shield2.json").exists():
                yield [PATCH, "scripts/eval_v5.py", "--shield2", g, a, str(s)], rd.parent / "eval_shield2.log", key
    if spec.get("classical"):
        # Non-learning baselines under the same protocol (no training; deterministic -> one run each).
        # planner_sf2 = the tuned Ackermann social force (gains selected on tuning episodes, seeds 5000+).
        from eval_classical_v5 import CONTROLLERS
        cdir = C.RUNS / "results" / g / "classical"
        cdir.mkdir(parents=True, exist_ok=True)
        for ctrl in CONTROLLERS:
            for shield in (False, True, "v2"):
                tag = f"{ctrl}{'' if not shield else ('_shield2' if shield == 'v2' else '_shield')}"
                if not (cdir / f"{tag}.json").exists():
                    flag = [] if not shield else (["--shield_v2"] if shield == "v2" else ["--shield"])
                    yield ([PATCH, "scripts/eval_classical_v5.py", g, ctrl, *flag], cdir / f"{tag}.log", g)


def run_all(g):
    # Per-run environment: PILOT_PIVOT / thread limits from the campaign plus the
    # run's own planner settings (a policy trained with a planner is evaluated with it).
    queue, procs = list(jobs(g)), []
    while queue or procs:
        while queue and len(procs) < PARALLEL:
            args, logf, key = queue.pop(0)
            procs.append(subprocess.Popen([PY, *args], cwd=str(ROOT), env=C.env_for(key, evaluation=True),
                                          stdout=open(logf, "a", encoding="utf-8"), stderr=subprocess.STDOUT))
        procs = [p for p in procs if p.poll() is None]
        time.sleep(5 if C.SMOKE else 15)


def aggregate(g):
    spec = C.GROUPS[g]
    out = C.RUNS / "results" / g
    out.mkdir(parents=True, exist_ok=True)
    table, per_task, per_seed = {}, {}, {}
    files = {"final": lambda rd: rd.parent / "eval_final.json", "mid": lambda rd: rd / "eval_mid.json",
             "best": lambda rd: rd / "eval_best.json", "shield": lambda rd: rd.parent / "eval_shield.json",
             "shield2": lambda rd: rd.parent / "eval_shield2.json"}
    for a in spec["agents"]:
        table[a], per_task[a], per_seed[a] = {}, {}, {}
        for tag, path in files.items():
            res = [json.load(open(path(run_dir(g, a, s))))["records"] for s in spec["seeds"]
                   if path(run_dir(g, a, s)).exists()]
            if len(res) < len(spec["seeds"]):
                continue
            rows = []
            for recs in res:
                m = lambda k: float(np.mean([r[k] for r in recs]))  # noqa: E731
                rows.append(dict(sr=m("success"), cvr=m("cvr"), spl=m("spl"), svr=m("svr"), mpd=m("mpd"),
                                 collision=float(np.mean([r["end"] == "collision" for r in recs])),
                                 timeout=float(np.mean([r["end"] == "timeout" for r in recs]))))
            table[a][tag] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            table[a][tag].update({f"{k}_std": float(np.std([r[k] for r in rows])) for k in rows[0]})
            per_seed[a][tag] = rows
            pooled = [r for recs in res for r in recs]
            per_task[a][tag] = {tk: dict(sr=float(np.mean([r["success"] for r in pooled if r["task_id"] == tk])),
                                         cvr=float(np.mean([r["cvr"] for r in pooled if r["task_id"] == tk])))
                                for tk in sorted({r["task_id"] for r in pooled})}
    for name, obj in (("table.json", table), ("per_task.json", per_task), ("per_seed.json", per_seed)):
        json.dump(obj, open(out / name, "w"), indent=2)
    print(f"{'agent':<12}{'ckpt':<8}{'SR %':>13}{'coll %':>8}{'tout %':>8}{'CVR':>7}{'SPL':>7}")
    for a in spec["agents"]:
        for tag, r in table[a].items():
            print(f"{a:<12}{tag:<8}{100*r['sr']:>7.1f} ±{100*r['sr_std']:>4.1f}{100*r['collision']:>8.1f}"
                  f"{100*r['timeout']:>8.1f}{r['cvr']:>7.3f}{r['spl']:>7.3f}")
    (out / "EVAL_DONE").write_text("ok\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    if sys.argv[1] in ("--shield", "--shield2"):
        shield_eval(sys.argv[2], sys.argv[3], int(sys.argv[4]), v2=(sys.argv[1] == "--shield2"))
    else:
        g = sys.argv[1]
        run_all(g)
        aggregate(g)
