"""Evaluate trained LMIC-HospitalNav checkpoints in Isaac Sim (Tier 2 ward).

This is an EVALUATION testbed, not a training environment. It loads a policy
that was trained in the 2-D kinematic simulator and drives the real MentorPi
articulation through the Tier-2 ward under PhysX, to see how the learned
behaviour survives real chassis dynamics.

    +---------------------+        +------------------------+
    | Isaac stage         |        | isaac_core (verified)  |
    |  pose, vel, scan    | -----> |  build_obs -> 90-D     |
    |  crowd positions    |        |  ackermann_joint_...   |
    +---------------------+        +------------------------+
              ^                                 |
              +------- joint targets -----------+

Everything numerically load-bearing lives in `isaac_core.py`, which is checked
bit-for-bit against the training environment (`python isaac_core.py`). This
file only moves data between Isaac and that core.

!! UNVERIFIED: this module imports Isaac Lab, which is not installed on the
   machine where it was written, so the API calls below have NOT been executed.
   Expect to adjust import paths and a few config field names for your exact
   Isaac Lab version. The physics and observation logic is the tested part.

    python deploy/isaac/eval_isaac.py --checkpoint <path> --episodes 20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

USD_WORLD = HERE / "mentorpi" / "hospital_tier2.usda"
URDF_ROBOT = HERE / "mentorpi" / "mentorpi_ackermann.urdf"
CFG = ROOT / "configs" / "hospital_v4.yaml"

GOAL_TOLERANCE = 0.6      # m, matches envs.lmic_hospital_env
MAX_STEPS = 1000          # 100 s at 10 Hz, matches the benchmark
CONTROL_HZ = 10.0

STEER_JOINTS = ["steer_lf_Joint", "steer_rf_Joint"]
DRIVE_JOINTS = ["wheel_lb_Joint", "wheel_rb_Joint"]   # rear-wheel drive


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="best_model.zip from runs_v4/<agent>_seed<n>_*/")
    ap.add_argument("--vecnorm", default=None,
                    help="matched vec_normalize pkl (REQUIRED for correct obs scaling)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--task", default="STAT_medicine")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--out", default=str(HERE / "results" / "isaac_eval.json"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Isaac must boot before any other Isaac import -------------------
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=args.headless).app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import UsdFileCfg

    from deploy.isaac.isaac_core import (
        ZoneMap, build_obs, ackermann_joint_targets, URDF_WHEEL_RADIUS,
    )
    from eval.evaluate import _load_model_and_env  # reuse the project's loader

    zones = ZoneMap(CFG)

    # ---- stage ----------------------------------------------------------
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / (CONTROL_HZ * 6), device="cuda:0"))
    sim_utils.UsdFileCfg(usd_path=str(USD_WORLD)).func("/World/Ward",
                                                       sim_utils.UsdFileCfg(usd_path=str(USD_WORLD)))

    # The URDF must be converted to USD once; see README. Point this at the
    # converted asset rather than re-importing the URDF every run.
    robot_usd = HERE / "mentorpi" / "mentorpi_ackermann.usd"
    if not robot_usd.exists():
        raise SystemExit(
            f"{robot_usd.name} not found. Convert the URDF first:\n"
            f"  see deploy/isaac/README.md, 'Convert the URDF'")

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=UsdFileCfg(usd_path=str(robot_usd)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(21.5, 15.0, 0.05)),
    )
    robot = Articulation(robot_cfg)
    sim.reset()

    steer_idx = [robot.find_joints(n)[0][0] for n in STEER_JOINTS]
    drive_idx = [robot.find_joints(n)[0][0] for n in DRIVE_JOINTS]

    # ---- policy ---------------------------------------------------------
    model, _, _, _ = _load_model_and_env(args.checkpoint, str(CFG))

    # ---- episodes -------------------------------------------------------
    goal = np.array(zones.fp.room_center("ward_icu_emerg"), dtype=float)
    records = []
    for ep in range(args.episodes):
        robot.reset()
        sim.reset()
        outcome, steps = "timeout", 0

        for steps in range(MAX_STEPS):
            root = robot.data.root_state_w[0].cpu().numpy()
            pos = root[:2]
            qw, qx, qy, qz = root[3:7]
            yaw = math.atan2(2 * (qw * qz + qx * qy),
                             1 - 2 * (qy * qy + qz * qz))
            speed = float(np.linalg.norm(root[7:9]))

            # TODO(isaac): attach a RayCaster/RTX lidar and read 48 rays here.
            # Until then this is a placeholder that reports free space, which
            # makes the policy behave as if the ward were empty.
            scan = np.full(48, 12.0, dtype=np.float32)

            obs = build_obs(pos=pos, yaw=yaw, speed=speed, lidar_ranges=scan,
                            people=np.zeros((0, 4)), goal=goal, zones=zones,
                            urgency=1.0, tod_hour=12.0)

            action, _ = model.predict(obs, deterministic=True)
            left, right, omega = ackermann_joint_targets(float(action[0]),
                                                         float(action[1]))

            robot.set_joint_position_target(
                np.array([[left, right]]), joint_ids=steer_idx)
            robot.set_joint_velocity_target(
                np.array([[omega, omega]]), joint_ids=drive_idx)
            robot.write_data_to_sim()
            for _ in range(6):                      # physics substeps per control tick
                sim.step()
            robot.update(1.0 / CONTROL_HZ)

            if np.linalg.norm(pos - goal) < GOAL_TOLERANCE:
                outcome = "reached"
                break
            if root[2] < 0.0 or root[2] > 0.5:      # fell through / flipped
                outcome = "unstable"
                break

        records.append({"episode": ep, "outcome": outcome, "steps": steps + 1})
        print(f"  ep {ep:3d}  {outcome:<9} {steps + 1:4d} steps")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "reached": sum(r["outcome"] == "reached" for r in records),
        "records": records,
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("Report these as per-policy outcomes (reached / not reached), NOT as a\n"
          "success rate comparable to the benchmark tables -- different simulator,\n"
          "different sensor model, different denominator.")
    app.close()


if __name__ == "__main__":
    main()
