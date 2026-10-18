# Gazebo Environment Spec — the "no-retraining" contract

You do **not** retrain. The trained policy is a fixed function `obs(90) -> [v,
steering]`. It transfers to Gazebo (and then real hardware) **as long as the
Gazebo world + robot + observation match the numbers below**. Match these and the
same `controller_node` runs unchanged; miss them and the policy sees a different
world than it trained on.

## 1. Map / world
| item | value | how in Gazebo |
|---|---|---|
| Floor size | 40 m x 30 m | `gen_gazebo_world.py` emits it from `configs/dmch_hospital.yaml` |
| Walls, beds, carts | the sim's `static_obstacles` | boxes (walls 2.0 m tall, beds/carts 0.6 m) |
| Cultural zones | prayer, male_ward, female_ward | coloured floor patches (visual) + the semantic PNG for context |
| Coordinate frame | x right, y up, origin at (0,0) | REP-103 (x fwd on the robot) |

## 2. Robot (Ackermann) — MUST match the sim kinematics
| constant | value |
|---|---|
| Wheelbase L | **0.20 m** |
| Body radius | 0.25 m |
| Linear speed v | **[-0.3, 0.8] m/s** |
| Steering delta | **[-30, +30] deg** ([-0.5236, 0.5236] rad) |
| Control rate | **10 Hz** (DT = 0.1 s) |
| Motion model | Ackermann bicycle: heading_rate = (v/L)*tan(delta) |

`hospitalbot.urdf.xacro` is tuned to this. Keep the Gazebo `ackermann_steering`
/ diff params close to the bicycle model (low inertia, minimal slip) so the robot
responds like the sim; large mismatch is the main sim-to-sim gap.

## 3. LiDAR — MUST match the LD19 profile the policy trained on
| constant | value |
|---|---|
| FOV | **360 deg** |
| Range max | **12.0 m** (obs normalized by this) |
| Rays fed to policy | **48** (downsampled by `scan_to_lidar48`) |
| Mount | facing forward (its 0 deg = robot +x) |

Gazebo may publish more beams (e.g. 360); `scan_to_lidar48` bins them to 48. Keep
`range_max = 12.0` and forward-facing mount.

## 4. Observation (90-D dict) — the exact contract
| block | dim | formula | Gazebo source |
|---|---|---|---|
| lidar | 48 | clip(range,0,12)/12 | Gazebo LiDAR -> /scan -> scan_to_lidar48 |
| robot_state | 5 | [x/40*2-1, y/30*2-1, sin, cos, v/0.8] | Gazebo odom / ground-truth pose |
| pedestrians | 20 | 5 x [dx/5, dy/5, vx/2.5, vy/2.5] (robot frame) | **ground truth** from actor poses (gt_pedestrians) |
| context | 14 | density5/30, urgency, zone_onehot(8), sin/cos(tod), density2/8, cult_dist/10 | semantic PNG + task |
| goal | 3 | [sin b, cos b, min(d/25,1)] (robot frame) | goal waypoint + pose |

Pedestrians are **easier** in Gazebo than on the real robot: Gazebo knows the
actor positions, so `gt_pedestrians.py` publishes `/people` directly — no detector.

## 5. Pedestrians (crowd)
| density | count | note |
|---|---|---|
| low / medium / high | 23 / 45 / 74 | spawn Gazebo `<actor>`s with scripted trajectories |
Types in the sim: medical_staff, patient, family_attendant, wheelchair. Behaviour
differs from the sim's HSFM crowd, but the policy reacts to their *positions* in
the obs, so it still responds.

## 6. Tasks + spawn + time
- Robot spawns near the nurse station; goal = a target room center (as in the sim).
- Time-of-day drives the prayer-window context; set `tod_hour` (12.0 default).
- Deploy `fixed_equal`/`attention` + shield for best success, or CWN + shield to
  show the live weight overlay.

## 7. What you accept as a gap (and do NOT retrain for)
- Real physics (inertia, friction, slip) vs the idealized bicycle model.
- Scripted actors vs HSFM pedestrians.
The reactive policy + shield tolerate these. Only if you want maximal real
transfer would you later train IN Gazebo with domain randomization (optional,
a separate effort) — not needed for a validation/demo.
