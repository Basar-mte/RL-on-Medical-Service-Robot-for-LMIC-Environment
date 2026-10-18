# Isaac_Package

Self-contained evaluation testbed: run **already-trained** LMIC-HospitalNav
policies through the Tier-2 ward in Isaac Sim, on the real MentorPi
articulation under PhysX. Nothing here trains, and no benchmark number in the
paper depends on it.

**Start with [WHAT_TO_DO.md](WHAT_TO_DO.md)** — a step-by-step walkthrough from
copying the folder to reporting results. This file is the reference; that one
is the procedure.

Copy this whole folder to the Isaac Sim machine and keep its internal structure
— the scripts locate the project root by relative depth.

## 1. Verify the assets survived the copy

Run these first, in the Isaac machine's Python (they need only numpy/pyyaml):

```bash
python deploy/isaac/check_urdf.py        # steering chain, meshes, mass
python deploy/isaac/check_world_usd.py   # stage vs. the YAML it came from
python deploy/isaac/isaac_core.py        # observation core vs. training env
```

All three must print a VERDICT line saying they match. If `isaac_core.py` fails,
stop: the observation assembly has drifted and any result would be meaningless.

## 2. Convert the URDF to USD (once)

In Isaac Sim, open the URDF importer and load
`deploy/isaac/mentorpi/mentorpi_ackermann.urdf` with:

- fixed base **off**
- joint drives: **position** for `steer_lf_Joint`, `steer_rf_Joint`;
  **velocity** for `wheel_lb_Joint`, `wheel_rb_Joint`
- self-collision **off**

Save the result as `deploy/isaac/mentorpi/mentorpi_ackermann.usd`, beside the
URDF. `eval_isaac.py` looks for it there and will say so if it is missing.

Sanity-check in the viewport before going further: the front wheels should
visibly steer when you drag the `steer_*_Joint` sliders, and the robot should
rest on its wheels rather than sink or bounce.

## 3. Load the ward

`deploy/isaac/mentorpi/hospital_tier2.usda` is generated from
`configs/hospital_v4.yaml` — the same file the RL environment rasterizes and
the Gazebo world is built from. Open it directly; it needs no conversion.

## 4. Run

```bash
python deploy/isaac/eval_isaac.py \
    --checkpoint models/fixed_equal_seed2_best.zip \
    --vecnorm    models/fixed_equal_seed2_vecnorm.pkl \
    --episodes 20
```

Always pass the **matched** vecnorm. A mismatched normalizer silently destroys
performance and looks like a policy failure.

## Before the numbers mean anything

`eval_isaac.py` has **not been executed** — it was written on a machine without
Isaac Lab. Two stubs must be finished, both marked in the file:

1. **LiDAR** is a placeholder reporting uniform free space, so the policy
   currently behaves as if the ward were empty. Attach a RayCaster or RTX lidar
   at 48 rays / 360 deg / 12 m and feed the ranges to `build_obs`.
2. **Crowd** is empty: the pedestrian block is zeros and both density terms
   read 0. Without a crowd this measures geometry and chassis dynamics only —
   a legitimate first experiment, but say so when reporting.

Expect to adjust some Isaac Lab import paths; that API moves between releases.

## Known geometry discrepancy

| | URDF (CAD) | `controller/ackermann.py` |
|---|---|---|
| Wheelbase | 0.1371 m | 0.145 m |
| Track | 0.1395 m | 0.133 m |

Isaac follows the URDF, the real robot follows the driver constant. Measure the
robot and fix whichever is wrong.

The URDF mass was also rescaled from the CAD export's 0.27 kg to **1.8 kg**,
which is an estimate, not a measurement. Weigh the robot and rebuild with
`python deploy/isaac/build_urdf.py --mass <kg>`.

## Reporting

Isaac's sensor model differs from the benchmark's exact ray-circle LiDAR, so the
observation distribution shifts even with identical geometry. Expect lower
numbers than Tier 2, report them as a **separate condition**, and use per-policy
outcomes — "reached" / "not reached" plus failure mode — never a success rate
placed beside the benchmark tables.
