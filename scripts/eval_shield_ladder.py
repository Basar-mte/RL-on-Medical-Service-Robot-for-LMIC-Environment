"""Shield ladder evaluation — the full proxemics/door experiment, eval-only.

Variants (all applied at inference to the SAME trained checkpoint):
  none      — raw policy
  western   — existing shield (stop 0.40 m, steer 1.2 m, full speed at 1.5 m)
  lmic      — LMIC-calibrated proxemics (stop 0.33 m, steer 0.7 m, full 0.9 m)
  door_lmic — lmic + stop-and-yield when blocked within 1.2 m of a doorway

Also reports SVR at BOTH 0.8 m (paper eq. 3 / Western personal space) and
0.4 m (LMIC operational spacing), computed here from true pedestrian
positions — no main-code changes anywhere (shield parameters are set by
monkeypatching baselines.safety_shield module constants).

    .venv\\Scripts\\python.exe scripts\\eval_shield_ladder.py
        [--models 1m|best] [--agents cwn,attention,vanilla] [--seeds 0,1,2]
        [--episodes 200] [--density medium] [--variants none,western,lmic,door_lmic]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy" / "ros2_stack" / "training"))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import baselines.safety_shield as ss  # noqa: E402
from envs.lmic_hospital_env import LMICHospitalNavEnv, TASK_IDS, LIDAR_MAX  # noqa: E402
from eval.evaluate import _batch, _unbatch, compute_spl  # noqa: E402
import layout  # noqa: E402  (the deploy layout — single source of door positions)

CFG = str(ROOT / "configs" / "hospital_v4.yaml")
RUNS = ROOT / "runs_v4"

# Shield parameter sets: (d_margin, full_speed_at, block)
VARIANTS = {
    "none":      None,
    "western":   (0.40, 1.5, 1.2),
    "lmic":      (0.33, 0.9, 0.7),
    "door_lmic": (0.33, 0.9, 0.7),
}
DOOR_NEAR = 1.2   # m — robot counts as "at a doorway" within this of a door center
DOOR_YIELD = 0.7  # m — frontal clearance below this while at a doorway => stop & yield

# Door centers in world coords, from the deploy layout (single source of truth)
DOORS = []
for (x0, y0, x1, y1, _name, doors) in layout.ROOMS:
    for wall, c in doors:
        DOORS.append({"S": (c, y0), "N": (c, y1),
                      "W": (x0, c), "E": (x1, c)}[wall])
DOORS = np.array(DOORS, dtype=float)


def set_shield(params):
    d_margin, full_at, block = params
    ss._D_MARGIN = d_margin
    ss._K = ss.MAX_LINEAR / (full_at - d_margin)
    ss._BLOCK = block


def find_dir(agent, seed):
    hits = [h for h in sorted(RUNS.glob(f"{agent}_seed{seed}_*")) if h.is_dir()]
    return hits[-1] if hits else None


def load_frozen(agent, seed, which):
    """Return (model, vecnorm_env, raw_env) for '1m' frozen or 'best' models."""
    d = find_dir(agent, seed)
    if d is None:
        raise FileNotFoundError(f"no run dir for {agent} seed{seed}")
    if which == "1m":
        zip_p, pkl_p = d / "final_model_1M.zip", d / "vec_normalize_1M.pkl"
    else:
        zip_p = d / "best_model" / "best_model.zip"
        pkl_p = d / "best_model" / "vec_normalize_best.pkl"
        if not pkl_p.exists():
            pkl_p = d / "vec_normalize.pkl"
    raw = LMICHospitalNavEnv(floorplan_path=CFG, use_camera=False)
    venv = DummyVecEnv([lambda: raw])
    vn = VecNormalize.load(str(pkl_p), venv)
    vn.training = False
    vn.norm_reward = False
    model = SAC.load(str(zip_p), device="auto")
    return model, vn, raw


def run_episodes(model, vn, raw, variant, n_episodes, density):
    params = VARIANTS[variant]
    if params:
        set_shield(params)
    recs = []
    for ep in range(n_episodes):
        task = TASK_IDS[ep % len(TASK_IDS)]
        vn.seed(1000 + ep)
        vn.set_options({"task": task, "density": density})
        obs = _unbatch(vn.reset())
        below08 = below04 = steps = 0
        mpd = float("inf")
        info = {}
        while True:
            a, _ = model.predict(_batch(obs), deterministic=True)
            a = np.asarray(a[0], dtype=np.float32)
            if params:
                raw01 = raw._cast_lidar()
                a = ss.shield_action(a, {"lidar": raw01})
                if variant == "door_lmic" and a[0] > 0:
                    d_front = float(np.asarray(raw01)[ss._FWD].min()) * LIDAR_MAX
                    near = np.hypot(*(DOORS - raw._robot_pos).T).min() < DOOR_NEAR
                    if near and d_front < DOOR_YIELD:
                        a = np.array([0.0, 0.0], dtype=np.float32)  # yield
            r, _, dones, infos = vn.step(a[np.newaxis])
            obs = _unbatch(r)
            info = infos[0]
            steps += 1
            if raw._crowd.agents:
                dmin = min(float(np.linalg.norm(p.pos - raw._robot_pos))
                           for p in raw._crowd.agents)
                mpd = min(mpd, dmin)
                below08 += dmin < 0.8
                below04 += dmin < 0.4
            if dones[0]:
                break
        succ = bool(info.get("task_complete", False))
        opt = info.get("path_length_optimal", 1.0)
        act = info.get("path_length_actual", opt)
        recs.append(dict(
            task_id=task, success=succ, steps=steps,
            collision=bool(info.get("collision", False)),
            cvr=info.get("cultural_violations", 0) / max(steps, 1),
            svr08=below08 / max(steps, 1), svr04=below04 / max(steps, 1),
            mpd=(0.0 if mpd == float("inf") else mpd),
            spl=compute_spl(succ, opt, act), ct_s=steps * 0.1))
    return recs


def agg(recs):
    m = lambda k: float(np.mean([r[k] for r in recs]))
    return dict(sr=m("success"), collision=m("collision"), cvr=m("cvr"),
                svr08=m("svr08"), svr04=m("svr04"), mpd=m("mpd"),
                spl=m("spl"), n=len(recs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="1m", choices=["1m", "best"])
    p.add_argument("--agents", default="cwn,attention,vanilla")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--density", default="medium")
    p.add_argument("--variants", default="none,western,lmic,door_lmic")
    args = p.parse_args()

    agents = args.agents.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")

    out = {}
    print(f"{'agent':<11}{'variant':<11}{'SR':>7}{'coll':>7}{'CVR':>8}"
          f"{'SVR@.8':>8}{'SVR@.4':>8}{'MPD':>7}{'SPL':>7}", flush=True)
    for agent in agents:
        for variant in variants:
            seed_aggs = []
            for seed in seeds:
                model, vn, raw = load_frozen(agent, seed, args.models)
                recs = run_episodes(model, vn, raw, variant,
                                    args.episodes, args.density)
                seed_aggs.append(agg(recs))
                vn.close()
            pooled = {k: float(np.mean([s[k] for s in seed_aggs]))
                      for k in seed_aggs[0] if k != "n"}
            pooled["sr_std"] = float(np.std([s["sr"] for s in seed_aggs]))
            pooled["n_seeds"] = len(seed_aggs)
            pooled["episodes_per_seed"] = args.episodes
            out[f"{agent}+{variant}"] = pooled
            print(f"{agent:<11}{variant:<11}{pooled['sr']:>7.3f}"
                  f"{pooled['collision']:>7.3f}{pooled['cvr']:>8.4f}"
                  f"{pooled['svr08']:>8.4f}{pooled['svr04']:>8.4f}"
                  f"{pooled['mpd']:>7.2f}{pooled['spl']:>7.3f}", flush=True)

    dest = RUNS / "results" / f"shield_ladder_{args.models}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
