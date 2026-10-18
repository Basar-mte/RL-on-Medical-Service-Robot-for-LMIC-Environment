# Isaac Sim evaluation testbed (Tier 2)

Run **existing** trained checkpoints through the Tier-2 ward in Isaac Sim, on
the real MentorPi articulation under PhysX. This is an evaluation testbed, not
a training environment: nothing here retrains, and no benchmark result is
affected by it.

```
deploy/isaac/
├── build_urdf.py        Hiwonder URDF -> physics-ready Ackermann URDF
├── check_urdf.py        validates the result before Isaac sees it
├── build_world_usd.py   configs/hospital_v4.yaml -> USD stage
├── check_world_usd.py   cross-checks the stage against that YAML
├── isaac_core.py        observation assembly + Ackermann conversion (TESTED)
├── eval_isaac.py        Isaac glue (UNTESTED — see below)
└── mentorpi/
    ├── mentorpi_ackermann.urdf + meshes/
    └── hospital_tier2.usda
```

## What is verified and what is not

| Component | Status |
|---|---|
| `mentorpi_ackermann.urdf` | ✅ single root, steering chain correct, 14/14 meshes resolve |
| `hospital_tier2.usda` | ✅ 92/92 obstacles match the YAML, nothing outside the map or below the floor |
| `isaac_core.py` | ✅ `robot_state`, `goal`, `context` bit-identical to the training env; zones 400/400 |
| `eval_isaac.py` | ⚠️ **never executed** — Isaac Lab is not installed on the authoring machine |

Run the checks any time:

```bash
python deploy/isaac/check_urdf.py
python deploy/isaac/check_world_usd.py
python deploy/isaac/isaac_core.py      # self-test against the training env
```

## Three things the URDF needed

Hiwonder's `ack.xacro` is a visualisation model, not a physics model.

1. **No steering joints.** All four wheels were `continuous` roll joints; the
   real robot steers through a servo linkage the driver handles in software, so
   the geometry was never modelled. `build_urdf.py` inserts revolute knuckles
   with ±45° limits (matching the clamp in `AckermannChassis.set_velocity`) and
   re-parents the front wheels onto them.
2. **Mass 6× too light.** The CAD export totals 0.27 kg for a robot carrying a
   Pi 5, battery and LiDAR. Invisible in RViz, decisive in PhysX. Rescaled to a
   default 1.8 kg — **weigh yours and pass `--mass`**.
3. **Noisy wheel axes** (`-2.29e-05 1 1.96e-05`) cleaned to exact `0 1 0`, and
   effort/velocity limits added so the joints are drivable.

### Geometry discrepancy — unresolved

| | URDF (CAD) | `controller/ackermann.py` |
|---|---|---|
| Wheelbase | 0.1371 m | 0.145 m |
| Track | 0.1395 m | 0.133 m |

8 mm and 6 mm apart. Isaac follows the URDF, the real robot follows the driver
constant, and the curvature remap divides by whichever you choose. Measure the
robot and correct whichever is wrong — they should not disagree.

## Setup

**Convert the URDF to USD** (once). In Isaac Sim, use the URDF importer with:
fixed base *off*; joint drives *position* for the two `steer_*_Joint`, *velocity*
for the `wheel_*_Joint`; self-collision *off*. Save as
`mentorpi/mentorpi_ackermann.usd` next to the URDF — `eval_isaac.py` expects it
there and will tell you if it is missing.

**Regenerate the world** whenever `hospital_v4.yaml` changes:

```bash
python deploy/isaac/build_world_usd.py && python deploy/isaac/check_world_usd.py
```

The stage is generated from the same YAML the RL environment rasterizes and the
Gazebo world is built from. Keep it that way — the paper's claim that training
geometry and deployment geometry cannot drift apart only holds while every
renderer reads one source.

## Finishing `eval_isaac.py`

Two things are stubbed and must be done before the numbers mean anything:

1. **LiDAR.** The scan is currently a placeholder reporting uniform free space,
   which makes the policy behave as if the ward were empty. Attach a
   `RayCasterCfg` (or RTX lidar) configured for 48 rays over 360° at 12 m, and
   feed the ranges to `build_obs`.
2. **Crowd.** `people` is empty, so the pedestrian block is zeros and both
   density terms read 0. Without a crowd this measures geometry and chassis
   dynamics only — which is a legitimate first experiment, but say so.

Also expect to adjust the Isaac Lab import paths and a few config field names;
that API moves between releases and these calls have not been run.

## Reporting

Isaac's sensor model differs from the benchmark's exact ray–circle LiDAR, so the
observation distribution shifts even with identical geometry. Expect lower
numbers than Tier 2, and report them as a **separate condition**, never merged
with or compared against `tab:wardb`.

Use per-policy outcomes — "reached" / "not reached" plus failure mode — not a
success rate. Different simulator, different sensor model, different
denominator; quoting a rate invites exactly the comparison a referee would flag.
