# ros_sim - Gazebo/RViz simulation and MentorPi deployment package

> **Public repository note.** This folder holds the ROS 2 package, the Gazebo plugins and worlds, the trained models, the run reports (`results_summary/`) and the numerical data (`paper_data/`, without the per-run pedestrian logs). The raw run folders, renders and videos named below (`results_runs*/`, `results_figures/`, `paper_figures/`, `results_superseded/`) are about 5 GB and are not included. They are available from the authors on request. The Hiwonder MentorPi meshes are not included either, see `hospitalnav_deploy/meshes/acker/README.md`.


Best seed of EVERY trained agent from the final campaign (both tiers) with the full
hierarchical stack: route planner (cultural costs) -> local driver -> kinematic
swept-path shield (v2). Built by `scripts/build_ros_sim.py`; file hashes and benchmark
numbers in `MANIFEST.json`.

## Quick start (Ubuntu 22.04, ROS 2 Humble, Gazebo classic)

```bash
mkdir -p ~/hn_ws/src && cp -r hospitalnav_deploy ~/hn_ws/src/
cd ~/hn_ws && colcon build --symlink-install && source install/setup.bash

# recommended driver (residual RL)
ros2 launch hospitalnav_deploy gazebo_full_bringup.launch.py \
    agent:=residual_sf cultural_ratio:=3.0 \
    model_zip:=$PWD/../models/residual_sf_s2/final_model.zip \
    vecnorm_pkl:=$PWD/../models/residual_sf_s2/vec_normalize.pkl

# classical driver - no model files at all
ros2 launch hospitalnav_deploy gazebo_full_bringup.launch.py agent:=classical_sf2 cultural_ratio:=3.0
```

To run ANY model in `models/`: set `agent:=` to its base name (drop the `t1_`/`mapless_`
prefix and `_sN` suffix; `risk_field` -> `agent:=he2025`), point `model_zip`/`vecnorm_pkl`
into its folder, and set `cultural_ratio` and (for td3/ppo) `algo:=` from the table below.

## Models in this package (Tier-2 benchmark: success %, v2 shield)

| folder | launch args | SR % | note |
|---|---|---|---|
| `residual_sf_s2` | `agent:=residual_sf` | **99.4** | recommended |
| (no model) | `agent:=classical_sf2` | 97.8 | tuned social force |
| `vanilla_s3` | `agent:=vanilla cultural_ratio:=0.0` | 96.1 | culture-blind - takes the prayer-room shortcut |
| `fixed_equal_s2` | `agent:=fixed_equal` | 95.4 | pure learned CARS policy |
| `cwn_s4` | `agent:=cwn` | 94.8 | learned reward weights |
| `nav_heavy_s4` | `agent:=nav_heavy` | 95.2 | nav-weighted |
| `attention_s4` | `agent:=attention` | 94.6 | attention weighter |
| `risk_field_s0` | `agent:=he2025 cultural_ratio:=0.0` | 95.7 | risk-field baseline |
| `safe_heavy_s2` | `agent:=safe_heavy` | 94.6 | safety-weighted |
| `td3_equal_s1` | `agent:=fixed_equal algo:=td3` | 93.2 (seed) | TD3 validation |
| `ppo_equal_s0` | `agent:=fixed_equal algo:=ppo` | 0 | does not navigate (negative result) |
| `mapless_fixed_equal_s3` | `agent:=fixed_equal` + goal_publisher | 63.4 (seed) | NO planner - routing-bottleneck demo |
| `mapless_vanilla_s2` | `agent:=vanilla` + goal_publisher | 59.4 (seed) | NO planner, culture-blind |

Tier-1 models (`t1_*`): same launch args as their Tier-2 namesakes. They were trained on
the OPEN tier; the shipped Gazebo world is the walled ward, so treat Tier-1 runs as
qualitative demos (the planner replans on the live map, so they do run).

Mapless models take the straight-line goal (publish the goal without the route_planner,
e.g. `ros2 run hospitalnav_deploy goal_publisher`); giving them the planner waypoint
would not match their training inputs.

## gz-sim Harmonic / ROS 2 Jazzy (the campaign actually run for the paper)

Gazebo Classic does not exist on Ubuntu 24.04, so the runs use gz-sim Harmonic
through `ros_gz`, with the REAL MentorPi Ackermann chassis (`urdf/mentorpi_real.urdf`:
vendor meshes and CAD link origins, wheelbase 0.137 m, LD19-spec lidar at z = 0.1535 m).

```bash
# one run (recommended driver, walled ward, 8 walking pedestrians, stall/wedge escape on)
ros2 launch hospitalnav_deploy gz_bringup.launch.py preset:=residual_sf tier:=walled \
    peds:=busy escape_enabled:=true gui:=true rviz:=true
# the campaign: 6 policies x 2 tiers x escape on/off, then the realism tiers
python3 sim/run_campaign.py --peds busy          # -> results_runs/campaign.md
python3 sim/run_campaign.py --peds crowd45       # the same with 45 walkers
# the statistics campaign: every configuration x pedestrian seeds 1-10 (480 runs, ~13 h)
python3 sim/run_campaign.py --peds busy,crowd45 --seeds 1-10 --out results_runs_repeats
python3 sim/analyze_repeats.py                   # success rates + Wilson CIs -> results_summary/gazebo_repeats.md
python3 sim/export_data.py                       # ALL numerical data -> paper_data/ (see its DATA_README.md)
python3 sim/plot_paths.py                        # -> results_figures/path_*.png
python3 sim/capture_figures.py --tier furnished --peds busy --robot 33,14.9,3.1416
```

**Worlds (`sim/gen_worlds.py`, collision geometry copied verbatim from the trained
`worlds/hospital.world`):**

| tier | what it is | scan-identical to training? |
|---|---|---|
| `walled` | Tier-2, all 40 walls, real beds, staff, doors and wall finishes (visibility split, see below) | yes - `verify_scan_invariance.py` |
| `open` | Tier-1, outer shell only, rooms are semantic zones | yes |
| `real` | identical to `walled` since realism v2 (every tier now carries the real assets); kept as an alias | yes |
| `furnished` | real + 97 floor-standing props WITH collision (bedside tables, chairs, IV stands, cabinets, X-ray, trolleys, wheelchairs, staff) that the prebuilt map does not contain, prayer mats | no, by design: unmapped-clutter stress test |

**Realism v2 (visibility split).** gz renders both the cameras and the LD19 lidar, and each
visual carries visibility flags that each sensor filters with a mask. Realism that must
reach the floor (door frames, open door leaves, the occupied patient beds, the staff) is
CAMERA-only; the trained shapes that would look wrong on camera (the 0.5 m bed boxes, the
standing people's cylinders) are LIDAR-only and keep their collision. Walls are one
full-height mesh each over the verbatim trained collision box, painted at physical scale
(two-tone, dado line, skirting); every doorway has a frame, a wall section above it, and
its door leaf open flat against the wall; signs sit on that wall section. Result: every
image shows a real ward, while the policy's 48-ray input matches the trained geometry to
0.14 mm (walled) and 0.21 mm (open), `verify_scan_invariance.py`. The robot's LD19 masks out
CAMERA-only visuals, and every figure and video camera masks out LIDAR-only ones. The
interactive gz GUI camera has no mask, so the GUI shows both layers.

**Pedestrians (`peds:=none|light|busy|crowd45`).** `light` = 4 and `busy` = 8 designed
walkers (nurse, doctor, visitor, patient with a cane) on closed loops through the corridor
and rooms (`hospitalnav_deploy/ped_routes.py`); `crowd45` adds 37 generated walkers
(`sim/gen_crowd_routes.py`: 20 corridor walkers on six lanes and 17 room visitors, every
loop planned on the walled map, each room entered through its own doorway). Every route
keeps >= 0.30 m from mapped obstacles (`sim/route_clearance.py`).

They are gz actors the LD19 sees (legs at the scan plane), so the deployment stack's own
LiDAR people tracker feeds the policy exactly as on the robot (`people_source:=lidar`;
`gt` feeds the simulated walker states instead). The `hospitalnav_gz_plugins`
PedestrianBodies system (a gz-sim system built in the same workspace) owns every walker:

- **Walking: social force** (Helbing & Molnar 1995). Each walker heads for a point 1 s
  ahead on its own loop and is pushed by relaxation toward its own pace, anisotropic
  repulsion from every other person, a stronger and wider one from the robot, and the
  walls of the tier's occupancy map. People keep their route but give each other and the
  robot room: they slow, stop and step around, and never enter a wall. Over 300 s of 45
  walkers this cuts touching pairs (< 0.44 m) from 5.9 to 0.8 at any instant, and pairs
  inside each other (< 0.30 m) from 3.0 to 0.07, against the fixed schedule.
  A walker never puts its body inside the robot's footprint: squeezed against the robot
  or bumped by it, it gives way (without this rule a walker squeezing past a stopped robot
  rolled it over in 2 of 480 runs; those runs are archived). `ped_yield:=false` restores
  the original non-yielding schedule. `ped_seed:=N` starts
  every walker at a different point of its loop (`ped_routes.phases`; 0 = loop start, the
  single-run campaign), so repeated runs meet different crowds.
- **Physical contact.** Each walker has a 70 kg collision cylinder, driven to the walker
  every physics step with a bounded force (the velocity command starts from the body's
  measured velocity and may change by at most 12 m/s^2 a second). So the robot cannot
  drive through a person, and a person shoves it - but cannot squeeze it out at speed
  or knock it over, which the first, infinitely stiff bodies did (`results_superseded/
  2026-10-17_stiff-pedestrian-bodies`). Bodies pass through each other (collision
  category bitmasks); the social force already keeps people apart.
- **Ground truth is what was simulated.** A walker that waits is no longer a function of
  sim time, so the plugin publishes the simulated poses (`/hospitalnav/pedestrians`,
  republished by `ped_gt` as `/pedestrian_poses`) and logs them per run to `peds.csv`.
  Every run record's proxemic outcomes come from them: closest approach, contacts
  (< 0.5 m centre distance = robot + body radius), physical touches (< 0.35 m) and seconds
  inside the 1.2 m personal zone. The run CSV also logs the chassis tilt, so a knock-down
  would show (none in the reported runs: max tilt 1.0 deg over all 555).

**Doorways.** The trained openings are 1.8 m wide (a bed-width double door), wider than
the layout's nominal 1.0 m door; frames, the wall section above each door and the open door
leaves are fitted to the trained opening, so no hollow is left beside a door. Wall-mounted
corridor items (clock, notice boards, extinguishers) sit on solid wall, clear of every
opening.

**Floors.** The textured floors are visual only, with their top surface at z = 0 - the physics
ground the robot's wheels, the walkers and the furniture stand on (until 2026-10-17 the floor
slabs' tops were at 0.02 m, so wheels, feet and furniture rendered 2 cm sunk into the tiles).

**Cultural outcome.** `sim/zone_metrics.py` reports, per run, the seconds, metres and
maximum depth the robot spent inside the prayer room and the gendered wards. In the
open tier the cultural route planner runs along the prayer zone's edge, so cultural
policies skim it by a few decimetres while culture-blind ones cross the room (~4 m deep).

**Pose source.** The Ackermann plugin's wheel odometry is dead reckoning and drifted
15-25 m in a 70 s test drive, so the stack navigates on gz's `OdometryPublisher`
(true pose; `map == odom`). Localisation is therefore ground truth - stated, not
implied - so a run measures the navigation policy, not localisation drift. The wheel
odometry is still published as `/odom_wheel` for the record.

**Clocks.** Goal budgets, stall/wedge timers and the run CSV are simulation time, so
outcomes do not depend on how fast the (unthrottled) simulator happened to run.

**Task protocol (`route:=direct`, the default).** One goal, the emergency room
(5, 7.5), from the nurse-desk spawn (37, 15); the trained route planner picks the
path on the prebuilt map and feeds the policy its look-ahead waypoint, as in
training. Budget 300 s sim (about 6x the nominal time of the 37 m route). The older
corridor-hop tour (`route:=hops`) is kept for floor demonstrations only: one of its
hops sits between the two standing people at (17.5, 15) and (20, 15), which made
every policy time out at the same spot.

**Rendering note.** On this machine every SDF `<pbr>` material renders black
(Intel iGPU, ogre2), so the generator emits textured surfaces as unit-cube OBJ
meshes with MTL textures (`meshes/gen/`), which render correctly; the collision
geometry is untouched.

**Outputs.** `paper_data/` holds every number of the simulation as CSV with a data dictionary
(`paper_data/DATA_README.md`): one row per run across all campaigns, the repeated campaign's
per-configuration statistics, every run's robot and pedestrian time series, the routes,
seeds, crowd statistics, door openings, scan-invariance results and model parameters.
`results_runs/campaign.{json,md}` (one record per run; each run directory holds
`run.csv`, `peds.csv`, `result.json`, `run.log`), `results_runs_realism/` (real + furnished
tiers), `results_figures/` (Gazebo renders per tier, `path_*.png` route figures per run,
`rviz_live_*.png`, `crowd_<tier>_<level>.csv` robot-free crowd logs from `sim/crowd_log.py`
that the stills freeze), `worlds/scan_invariance_*.json` (appendix numbers), and
`results_summary/gazebo_campaign.md` from `python3 sim/make_report.py`. Media:
`sim/make_publication_media.sh` - the demo videos are the run's live RViz mapping stream
over Gazebo footage re-rendered from the run's own logs (`sim/replay_video.py`: the
`ReplayRig` system poses the robot and camera, PedestrianBodies replays `peds.csv`, all
inside the simulator, so no frame is lost or shows stale poses).

## RViz

Add `/map`, `/rl_route` (check it avoids the prayer room before the robot moves),
`/rl_goal`, `/scan`, `/pedestrian_markers`, `/sim_markers`, TF. Goals: RViz
"2D Goal Pose", or the patrol sequencer runs the nurse-desk -> emergency-room task
automatically.

## Key parameters

| param | default | meaning |
|---|---|---|
| `agent` | - | driver mode (see table) |
| `algo` | sac | SB3 class of model_zip: sac / td3 / ppo |
| `cultural_ratio` | 3.0 | planner zone costs; **0.0 for vanilla / risk_field** (trained on the geo planner) |
| `shield_version` | 2 | 2 = swept-path (zero collisions in benchmark); 1 = forward cone |
| `residual_scale` | 0.5 | residual agents only - must match training |
| `lookahead` | 2.5 | planner waypoint distance (trained value - do not change) |

**Always keep each model with its own `vec_normalize.pkl`** - a mismatched normaliser
silently destroys performance.

## Real robot (MentorPi)

Follow `deploy/MENTORPI_DEPLOYMENT.md` in the repository: manual SLAM drive -> zone-label
PNG -> localization -> autonomous runs. The command bridge keeps the TRAINING wheelbase
(0.20 m) so commanded curvature is preserved.

## Reporting convention (important)

Report physical/Gazebo runs as per-policy binary outcomes - "reached" / "not reached"
plus the failure mode, or "reached in N of M attempts". **Never synthesize a success
rate from demonstration runs.**

## Known limits (measured in the benchmark)

* A non-yielding crowd degrades every driver at this density; the shield cannot prevent
  someone walking into the robot.
* Unmapped obstacles are handled (zero collisions with the v2 shield) at the cost of
  timeouts; replanning follows the live SLAM map.
* The zero-collision result assumes clean actuation: 10 % motor noise yields
  1.4-3.0 % contacts in simulation.
