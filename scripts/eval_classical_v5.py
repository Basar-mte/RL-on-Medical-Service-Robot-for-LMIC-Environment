"""Classical (non-learning) baselines for the final campaign, evaluated with the
paper protocol (deterministic, episode seed 1000+ep, tasks cycled, N episodes):

  planner_sf   route planner (cultural, 3x — the trained hierarchy's global level)
               + SocialForceController steering to the planner's waypoint   = "the stack without RL"
  planner_pp   route planner + pure-pursuit waypoint follower (no pedestrian avoidance)
  astar_pp     baselines.classical.AStarController: A* path at episode start + pure pursuit
  sf           SocialForceController toward the FINAL goal (reactive, no global plan)

Each with --shield for the inference safety shield. Output (same record format as
eval/evaluate.py): runs_v5/results/<group>/classical/<ctrl>[_shield].json

    python scripts/eval_classical_v5.py <group> <ctrl> [--shield] [--n_eps N]
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "experiments"))
import campaign_v5_stable as C  # noqa: E402
from envs.lmic_hospital_env import LMICHospitalNavEnv, TASK_IDS, K_NEIGHBORS  # noqa: E402
from baselines.classical import AStarController, SocialForceController, _omega_to_steer  # noqa: E402
from baselines.safety_shield import shield_action  # noqa: E402
from planner.arc_shield import shield_action as shield_v2  # noqa: E402
from eval.evaluate import compute_spl  # noqa: E402
from planner.planner import Planner  # noqa: E402

CONTROLLERS = ["planner_sf", "planner_pp", "astar_pp", "sf", "planner_sf2", "sf2", "planner_orca"]
# sf2 = the TUNED Ackermann social force WITHOUT the route planner (goal force aims at the
# final destination): isolates what the planner contributes for the classical driver.
# planner_orca = ORCA (velocity obstacles + LP; experiments/planner/orca.py) to the waypoint.
PLANNER_L, PLANNER_RATIO = float(C.PLANNER_DEFAULTS["PLANNER_L"]), float(C.PLANNER_DEFAULTS["PLANNER_RATIO"])


def pure_pursuit(pos, heading, target, max_v=0.8):
    d = target - pos
    err = (math.atan2(d[1], d[0]) - heading + math.pi) % (2 * math.pi) - math.pi
    omega = float(np.clip(2.0 * err, -math.pi, math.pi))
    v = max_v * max(0.0, 1.0 - abs(err) / math.pi)
    return np.array([v, _omega_to_steer(omega, v)], dtype=np.float32)


from planner.sf_prior import AckermannSF  # noqa: E402  (Ackermann-aware social force; shared with residual RL)


def run(group, ctrl, shield, n_eps, density="medium", gw=1.5, ww=3.0, vmin=0.25, vmax=0.8, suffix="",
        seed_base=1000, tau=2.0):
    """seed_base: episode seeds are seed_base+ep. The protocol uses 1000 (= the trained agents'
    evaluation episodes); gain sweeps use a DIFFERENT base (e.g. 5000) so the configuration is
    selected on tuning episodes and scored on the protocol ones."""
    cfg = os.environ.get("EVAL_CONFIG") or C.GROUPS[group]["config"]   # EVAL_CONFIG: world override (unmapped obstacles)
    import planner.uncertainty as unc
    unc.apply_from_env()     # UNC_* robustness patches (no-op without the env vars)
    env = LMICHospitalNavEnv(floorplan_path=str(ROOT / cfg), use_camera=False)
    fp_plan = env.fp
    if os.environ.get("PLANNER_MAP"):      # stale-map planning: the planner cannot see the override world
        from envs.floorplan import load_floorplan
        fp_plan = load_floorplan(os.environ["PLANNER_MAP"])
    planner = Planner(fp_plan, ratio=PLANNER_RATIO) if ctrl.startswith("planner") else None
    astar = AStarController()
    if ctrl in ("planner_sf2", "sf2") and (gw, ww, vmin, vmax) == (1.5, 3.0, 0.25, 0.8):
        sf = AckermannSF.tuned()        # CLI defaults untouched -> the tuned gains (seeds 5000+ selection)
    elif ctrl in ("planner_sf2", "sf2"):
        sf = AckermannSF(vmin=vmin, goal_weight=gw, wall_weight=ww, max_linear=vmax)
    else:
        sf = SocialForceController()
    orca = None
    if ctrl == "planner_orca":
        from planner.orca import OrcaController
        orca = OrcaController(tau=tau, vmin=vmin if vmin != 0.25 else 0.4, v_pref=vmax)
    recs, t0 = [], time.time()
    for ep in range(n_eps):
        task = TASK_IDS[ep % len(TASK_IDS)]
        obs, info = env.reset(seed=seed_base + ep, options={"task": task, "density": density})
        if ctrl == "astar_pp":
            astar.plan(env.fp, env._robot_pos, env._task_goal)
        if ctrl in ("sf", "sf2"):
            sf.set_goal(env._task_goal)     # mapless: the goal force aims at the FINAL destination
        while True:
            pos, heading = env._robot_pos, env._robot_heading
            if planner is not None:
                wp = planner.field(env._task_goal, env._tod_hour).waypoint(pos, PLANNER_L, True)
                target = env._task_goal if wp is None else wp
            if ctrl == "planner_orca":
                orca.set_goal(np.asarray(target, float))
                act = orca.act(pos, heading, env._robot_speed, obs["pedestrians"].reshape(K_NEIGHBORS, 4),
                               obs["lidar"])
            elif ctrl in ("planner_sf", "planner_sf2"):
                sf.set_goal(np.asarray(target, float))
                act = sf.act(pos, heading, obs["pedestrians"].reshape(K_NEIGHBORS, 4), obs["lidar"])
            elif ctrl == "planner_pp":
                act = pure_pursuit(pos, heading, np.asarray(target, float))
            elif ctrl == "astar_pp":
                act = astar.act(pos, heading)
            else:
                act = sf.act(pos, heading, obs["pedestrians"].reshape(K_NEIGHBORS, 4), obs["lidar"])
            if shield == "v2":
                act = shield_v2(act, {"lidar": obs["lidar"], "pedestrians": obs["pedestrians"],
                                      "robot_state": obs["robot_state"]})
            elif shield:
                act = shield_action(act, {"lidar": obs["lidar"]})
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                steps = max(int(info["steps"]), 1)
                succ, coll = bool(info["task_complete"]), bool(info["collision"])
                opt = info.get("path_length_optimal", 1.0); actual = info.get("path_length_actual", opt)
                recs.append(dict(ep=ep, task_id=task, success=succ,
                                 end="success" if succ else ("collision" if coll else "timeout"), steps=steps,
                                 dist_to_goal=float(info["dist_to_goal"]), cvr=info["cultural_violations"] / steps,
                                 svr=info["social_violations"] / steps, spl=compute_spl(succ, opt, actual),
                                 mpd=float(info["min_pedestrian_distance"]), ct_s=steps * 0.1))
                break
        if (ep + 1) % 50 == 0:
            print(f"  {ctrl}{'+shield' if shield else ''} ep {ep + 1}/{n_eps} SR {np.mean([r['success'] for r in recs]):.3f}"
                  f" ({time.time() - t0:.0f} s)", flush=True)
    env.close()
    stag = "" if not shield else ("_shield2" if shield == "v2" else "_shield")
    out = C.RUNS / "results" / group / "classical" / f"{ctrl}{suffix}{stag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(sr=float(np.mean([r["success"] for r in recs])),
                   collision=float(np.mean([r["end"] == "collision" for r in recs])),
                   timeout=float(np.mean([r["end"] == "timeout" for r in recs])), n=len(recs))
    json.dump(dict(group=group, controller=ctrl, shield=shield, planner=dict(L=PLANNER_L, ratio=PLANNER_RATIO),
                   gains=(dict(goal_weight=gw, wall_weight=ww, vmin=vmin, vmax=vmax) if "sf2" in ctrl
                          else dict(tau=tau, vmin=vmin, v_pref=vmax) if ctrl == "planner_orca" else None),
                   seed_base=seed_base, summary=summary, records=recs), open(out, "w"), indent=1)
    print(f"[classical] {group} {ctrl}{'+shield' if shield else ''}: {summary} -> {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group"); p.add_argument("ctrl", choices=CONTROLLERS)
    p.add_argument("--shield", action="store_true")
    p.add_argument("--shield_v2", action="store_true", help="kinematic swept-path shield (experiments/planner/arc_shield.py)")
    p.add_argument("--n_eps", type=int, default=3 if C.SMOKE else 500)
    p.add_argument("--gw", type=float, default=1.5, help="planner_sf2 goal weight")
    p.add_argument("--ww", type=float, default=3.0, help="planner_sf2 wall weight")
    p.add_argument("--vmin", type=float, default=0.25, help="planner_sf2 minimum speed fraction while turning")
    p.add_argument("--vmax", type=float, default=0.8, help="planner_sf2 max speed (m/s)")
    p.add_argument("--tau", type=float, default=2.0, help="planner_orca time horizon (s)")
    p.add_argument("--suffix", default="", help="output-name suffix (gain sweeps)")
    p.add_argument("--seed_base", type=int, default=1000, help="episode seed base (1000 = protocol; sweeps use another)")
    a = p.parse_args()
    run(a.group, a.ctrl, "v2" if a.shield_v2 else a.shield, a.n_eps, gw=a.gw, ww=a.ww, vmin=a.vmin, vmax=a.vmax,
        suffix=a.suffix, seed_base=a.seed_base, tau=a.tau)
