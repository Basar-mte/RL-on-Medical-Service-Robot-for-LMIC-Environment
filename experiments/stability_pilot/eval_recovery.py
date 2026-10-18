"""Shield + RECOVERY behaviour, evaluation only (no retraining).

stuck_anatomy.py showed the shielded robot's timeouts are 97-100% "frozen"
(moved <1 m in the last 10 s), mostly NOT pedestrian-blocked (ped within 1 m
in only 13-14%), inside wards/prayer room/corridor facing furniture or walls:
the policy keeps pushing toward the goal bearing, the shield caps forward
speed to ~0, and the Ackermann chassis cannot turn in place -> deadlock. The
policy never learned recovery because in training "blocked" meant "crashed".

Recovery (a Nav2-style "back up and turn" rule wrapped around the shield):
if the robot moved < STUCK_DIST over the last STUCK_STEPS steps, reverse for
REV_STEPS with full steering chosen to swing the NOSE toward the clearer side
(bicycle model: reversing with steer delta turns the heading by -sign(delta)),
then hand back to the policy (+ shield) with a cooldown. If the rear is
blocked, skip and let the shield handle it.

    python experiments/stability_pilot/eval_recovery.py <agent> <seed> <n_eps>
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from eval.evaluate import _load_model_and_env, _batch, _unbatch  # noqa: E402
from envs.lmic_hospital_env import (TASK_IDS, LIDAR_MAX, LIDAR_FOV, N_LIDAR_RAYS,  # noqa: E402
                                    MIN_LINEAR, MAX_STEER, lidar_ray_angles)
from baselines.safety_shield import shield_action  # noqa: E402

STUCK_STEPS, STUCK_DIST = 30, 0.30   # 3 s, 0.3 m
REV_STEPS, COOLDOWN = 15, 20         # 1.5 s reverse, then >= 2 s of policy control
REAR_MIN = 0.45                       # m clearance needed behind to reverse

ANG = np.degrees(lidar_ray_angles(LIDAR_FOV, N_LIDAR_RAYS))
LEFT = (ANG > 30) & (ANG <= 120)
RIGHT = (ANG < -30) & (ANG >= -120)
REAR = np.abs(ANG) >= 150

a, s, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rd = sorted(p for p in (HERE / "runs4" / f"{a}_s{s}").glob(f"{a}_seed{s}_*") if p.is_dir())[-1]
model, inner, env, _ = _load_model_and_env(str(rd), "configs/hospital_v4.yaml")

recs, n_recov = [], 0
for ep in range(n):
    task = TASK_IDS[ep % len(TASK_IDS)]
    env.seed(1000 + ep); env.set_options({"task": task, "density": "medium"})
    obs = _unbatch(env.reset())
    trail, rev_left, cool = [], 0, 0
    while True:
        lid = np.asarray(inner._cast_lidar(), float)
        pos = np.array(inner._robot_pos, float); trail.append(pos)
        if rev_left > 0:
            rev_left -= 1
            act = np.array([MIN_LINEAR, recov_steer], dtype=np.float32)
        else:
            cool = max(cool - 1, 0)
            stuck = (len(trail) > STUCK_STEPS and cool == 0
                     and np.linalg.norm(trail[-1] - trail[-1 - STUCK_STEPS]) < STUCK_DIST)
            rear_free = lid[REAR].min() * LIDAR_MAX > REAR_MIN
            if stuck and rear_free:
                left_clear, right_clear = lid[LEFT].min(), lid[RIGHT].min()
                # reversing: heading_rate = (v/L) tan(delta) with v<0, so to swing the
                # nose LEFT (heading increases) we need delta < 0, and vice versa.
                recov_steer = -MAX_STEER if left_clear > right_clear else MAX_STEER
                rev_left, cool, n_recov = REV_STEPS - 1, REV_STEPS + COOLDOWN, n_recov + 1
                act = np.array([MIN_LINEAR, recov_steer], dtype=np.float32)
            else:
                p_act, _ = model.predict(_batch(obs), deterministic=True)
                act = shield_action(p_act[0], {"lidar": lid})
        raw, _, dones, infos = env.step(act[np.newaxis]); obs = _unbatch(raw); info = infos[0]
        if dones[0]:
            steps = max(info.get("steps", 1), 1)
            succ = bool(info.get("task_complete")); coll = bool(info.get("collision"))
            recs.append(dict(task_id=task, success=succ, end="success" if succ else ("collision" if coll else "timeout"),
                             cvr=info.get("cultural_violations", 0) / steps))
            break
env.close()

json.dump({"records": recs, "n_recoveries": n_recov}, open(HERE / f"recovery_{a}_s{s}.json", "w"))
N = len(recs); ends = collections.Counter(r["end"] for r in recs)
stat = [r for r in recs if r["task_id"] == "STAT_medicine"]
print(f"=== {a}_s{s} shield+recovery, {N} eps: " + ", ".join(f"{k} {100*v/N:.0f}%" for k, v in ends.most_common())
      + f" | CVR {np.mean([r['cvr'] for r in recs]):.3f} | STAT {100*np.mean([r['success'] for r in stat]):.0f}%/"
      f"{np.mean([r['cvr'] for r in stat]):.3f} | recoveries {n_recov} ({n_recov/N:.1f}/ep)")
