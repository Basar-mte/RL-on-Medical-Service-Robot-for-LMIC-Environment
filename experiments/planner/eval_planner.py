"""Evaluate a trained policy WITH the route planner in the loop (evaluation-only:
the policy is unchanged; only its goal observation is steered by the planner).

    python experiments/planner/eval_planner.py --run_dir <run dir> --mode cult \
        [--ratio 1.0 --L 2.5 --dist euclid --no_los --shield --n_eps 200 \
         --config configs/hospital_v4.yaml --density medium] --out results/x.json

Protocol = eval/evaluate.py (deterministic, episode seed 1000+ep, tasks cycled),
plus the inference safety shield when --shield. Per-episode records keep the end
reason, zone step counts (prayer / prayer-in-window / gender wards) and, for
timeouts, whether the robot was frozen (<1 m moved in its last 10 s).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--config", default="configs/hospital_v4.yaml")
    p.add_argument("--mode", default="cult", choices=["off", "geo", "cult"])
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--L", type=float, default=2.5)
    p.add_argument("--dist", default="euclid", choices=["euclid", "geo"])
    p.add_argument("--no_los", action="store_true")
    p.add_argument("--shield", action="store_true")
    p.add_argument("--shield_v2", action="store_true", help="kinematic swept-path shield (arc_shield.py)")
    p.add_argument("--n_eps", type=int, default=200)
    p.add_argument("--density", default="medium")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import torch
    torch.set_num_threads(1)
    import planner.uncertainty as unc
    unc.apply_from_env()     # UNC_* robustness patches (no-op without the env vars)
    if a.mode != "off":
        from planner.patch import apply_planner_patch
        apply_planner_patch(mode=a.mode, ratio=a.ratio, lookahead=a.L, dist=a.dist,
                            progress=False, los=not a.no_los)
    from eval.evaluate import _load_model_and_env, _batch, _unbatch, compute_spl
    from envs.lmic_hospital_env import TASK_IDS
    from baselines.safety_shield import shield_action
    if a.shield_v2:
        from planner.arc_shield import shield_action as shield_v2
        a.shield = True

    model, inner, env, _ = _load_model_and_env(a.run_dir, a.config)
    residual = bool(getattr(type(inner), "_residual_patched", False))   # shield applied in-env to the executed action
    if residual and a.shield:
        inner._env_shield = "v2" if a.shield_v2 else "v1"
    term = {}
    _orig_reset = inner.reset

    def _stash(*args, **kw):          # terminal state before the VecEnv auto-reset
        if hasattr(inner, "_crowd"):
            term["pos"] = np.array(inner._robot_pos, float).copy()
            term["zone"] = str(inner.fp.get_zone(*inner._robot_pos))
        return _orig_reset(*args, **kw)
    inner.reset = _stash

    recs, t0 = [], time.time()
    for ep in range(a.n_eps):
        task = TASK_IDS[ep % len(TASK_IDS)]
        env.seed(1000 + ep); env.set_options({"task": task, "density": a.density})
        obs = _unbatch(env.reset())
        trail, prayer, prayer_win, gender = [], 0, 0, 0
        while True:
            act, _ = model.predict(_batch(obs), deterministic=True)
            if residual:
                act = act[0]
            elif a.shield_v2:
                act = shield_v2(act[0], {"lidar": inner._cast_lidar(), "pedestrians": inner._ped_obs_vec(),
                                         "robot_state": inner._robot_state_vec()})
            elif a.shield:
                act = shield_action(act[0], {"lidar": inner._cast_lidar()})
            else:
                act = act[0]
            pos = np.array(inner._robot_pos, float).copy(); trail.append(pos)
            z = inner.fp.get_zone(pos[0], pos[1])
            if z == "prayer":
                prayer += 1; prayer_win += int(inner.fp.is_prayer_window(inner._tod_hour))
            elif z in ("male_ward", "female_ward"):
                gender += 1
            raw, _, dones, infos = env.step(act[np.newaxis]); obs = _unbatch(raw); info = infos[0]
            if dones[0]:
                steps = max(int(info.get("steps", 1)), 1)
                succ = bool(info.get("task_complete")); coll = bool(info.get("collision"))
                opt = info.get("path_length_optimal", 1.0); actual = info.get("path_length_actual", opt)
                last = np.array(trail[-100:])
                recs.append(dict(
                    ep=ep, task_id=task, success=succ,
                    end="success" if succ else ("collision" if coll else "timeout"), steps=steps,
                    cvr=info.get("cultural_violations", 0) / steps, svr=info.get("social_violations", 0) / steps,
                    spl=compute_spl(succ, opt, actual), mpd=float(info.get("min_pedestrian_distance", 0.0)),
                    dist_to_goal=float(info.get("euclid_dist_to_goal", info.get("dist_to_goal", -1.0))),
                    prayer_steps=prayer, prayer_window_steps=prayer_win, gender_steps=gender,
                    end_zone=term.get("zone"),
                    frozen=bool(len(last) > 1 and np.linalg.norm(last[-1] - last[0]) < 1.0) if not succ and not coll else False))
                break
        if (ep + 1) % 20 == 0:
            sr = np.mean([r["success"] for r in recs])
            print(f"  ep {ep + 1}/{a.n_eps}  SR {sr:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    env.close()

    m = lambda k: float(np.mean([r[k] for r in recs]))  # noqa: E731
    summary = dict(sr=m("success"), collision=float(np.mean([r["end"] == "collision" for r in recs])),
                   timeout=float(np.mean([r["end"] == "timeout" for r in recs])), cvr=m("cvr"), svr=m("svr"),
                   spl=m("spl"), mpd=m("mpd"), prayer_cvr=float(np.mean([r["prayer_steps"] / r["steps"] for r in recs])),
                   frozen=float(np.mean([r["frozen"] for r in recs])), n=len(recs),
                   per_task={t: (float(np.mean(v)) if v else None)
                             for t in TASK_IDS for v in [[r["success"] for r in recs if r["task_id"] == t]]})
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(a), summary=summary, records=recs), open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"[eval_planner] saved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
