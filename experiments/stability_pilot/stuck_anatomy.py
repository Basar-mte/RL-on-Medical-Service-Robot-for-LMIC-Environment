"""Where does the shielded robot get stuck? For timeouts (and collisions) of a
shielded final policy, record the terminal state: zone, distance to nearest
pedestrian / doorway, pedestrians within 2 m, and how far the robot moved in
its last 100 steps (stuck vs wandering). Terminal state is captured from a
patched inner reset (the VecEnv auto-resets inside step()).

    python experiments/stability_pilot/stuck_anatomy.py <agent> <seed> <n_eps>
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy" / "ros2_stack" / "training"))
from eval.evaluate import _load_model_and_env, _batch, _unbatch  # noqa: E402
from envs.lmic_hospital_env import TASK_IDS  # noqa: E402
from baselines.safety_shield import shield_action  # noqa: E402
import layout  # noqa: E402

a, s, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rd = sorted(p for p in (HERE / "runs4" / f"{a}_s{s}").glob(f"{a}_seed{s}_*") if p.is_dir())[-1]
DOORS = np.array([{"S": (c, y0), "N": (c, y1), "W": (x0, c), "E": (x1, c)}[w]
                  for (x0, y0, x1, y1, _n, ds) in layout.ROOMS for w, c in ds], dtype=float)

model, inner, env, _ = _load_model_and_env(str(rd), "configs/hospital_v4.yaml")
TERM = {}
_orig_reset = inner.reset
def _stash(*args, **kw):
    if hasattr(inner, "_crowd"):
        TERM["pos"] = np.array(inner._robot_pos, float).copy()
        TERM["peds"] = [np.array(p.pos, float).copy() for p in inner._crowd.agents]
        TERM["zone"] = str(inner.fp.get_zone(*inner._robot_pos))
        TERM["goal_d"] = float(np.linalg.norm(inner._robot_pos - inner._task_goal))
    return _orig_reset(*args, **kw)
inner.reset = _stash

rows, ends = [], collections.Counter()
for ep in range(n):
    task = TASK_IDS[ep % len(TASK_IDS)]
    env.seed(1000 + ep); env.set_options({"task": task, "density": "medium"})
    obs = _unbatch(env.reset()); trail = []
    while True:
        act, _ = model.predict(_batch(obs), deterministic=True)
        act = shield_action(act[0], {"lidar": inner._cast_lidar()})
        trail.append(np.array(inner._robot_pos, float).copy())
        raw, _, dones, infos = env.step(act[np.newaxis]); obs = _unbatch(raw); info = infos[0]
        if dones[0]:
            end = "success" if info.get("task_complete") else ("collision" if info.get("collision") else "timeout")
            ends[end] += 1
            if end != "success":
                p = TERM["pos"]; peds = TERM["peds"]
                pd = np.array([np.linalg.norm(p - q) for q in peds]) if peds else np.array([99.0])
                last = np.array(trail[-100:])
                rows.append(dict(task=task, end=end, zone=TERM["zone"], goal_d=TERM["goal_d"],
                                 ped_d=float(pd.min()), peds_2m=int((pd < 2.0).sum()),
                                 door_d=float(np.min(np.linalg.norm(DOORS - p, axis=1))),
                                 moved_last100=float(np.linalg.norm(last[-1] - last[0])) if len(last) > 1 else 0.0,
                                 x=float(p[0]), y=float(p[1])))
            break
env.close()

out = HERE / f"stuck_{a}_s{s}.json"
json.dump({"ends": dict(ends), "rows": rows}, open(out, "w"), indent=1)
N = sum(ends.values())
print(f"\n=== {a}_s{s} + shield, {N} eps: " + ", ".join(f"{k} {100*v/N:.0f}%" for k, v in ends.most_common()))
to = [r for r in rows if r["end"] == "timeout"]
if to:
    R = lambda k: np.array([r[k] for r in to])
    print(f"TIMEOUTS ({len(to)}): pedestrian within 1 m {np.mean(R('ped_d') < 1.0):.0%} | >=2 peds within 2 m {np.mean(R('peds_2m') >= 2):.0%} | "
          f"within 1.5 m of a doorway {np.mean(R('door_d') < 1.5):.0%} | moved <1 m in last 10 s (frozen) {np.mean(R('moved_last100') < 1.0):.0%} | "
          f"median distance left to goal {np.median(R('goal_d')):.1f} m")
    print("  zones:", {k: f"{v/len(to):.0%}" for k, v in collections.Counter(r['zone'] for r in to).most_common(6)})
    print("  tasks:", dict(collections.Counter(r['task'] for r in to)))
