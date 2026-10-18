# Deploying LMIC-HospitalNav on the Hiwonder MentorPi

End-to-end process for running the trained RL policies on the physical robot:
**manual-drive mapping → localization → autonomous start-to-goal runs.**

This supersedes the platform-specific parts of `ros2_stack/DEPLOYMENT_GUIDE.md`,
which was written for a Yahboom RRC-Lite board and does not apply to this robot.

---

## 1. The one architectural decision that matters

The policies are **hierarchical** (October 2026 campaign): a global route planner
picks the route on the map, the RL policy drives it — crowd, doorways, timing —
and the safety shield filters every command. The wiring on the robot must be the
wiring they were trained with:

| | Mapless policies (pre-Oct-2026, `*_mapless` baselines) | **Hierarchical policies (current)** |
|---|---|---|
| Global route chosen by | the RL policy itself | `route_planner` (same cost field as training) |
| `/rl_goal` carries | the **final** destination | waypoint **2.5 m along the route**; `z` = remaining route length |
| Goal channel value | straight-line distance ÷ 25 | **route length ÷ 25** (the controller uses `z`) |
| Cultural routing | learned (fragile, 46 % timeouts) | planner cost: prayer room ×(1 + 3·c), gender wards mild |

**Use `route_planner` + `controller` for every current policy.** Feeding the final
destination to a hierarchical policy, or a planner waypoint to a mapless policy,
is a train/deploy mismatch.

### Why the hierarchy

The mapless policies froze: with the shield, 46 % of episodes timed out, 97–100 %
of those with the robot stuck behind walls or beds and no pedestrian within 1 m.
The route planner removed that (evaluation-only test on the same policies: 41.5 %
→ 83.6 % success, timeouts 46 % → 7.5 %), and the campaign retrained every agent
with the planner in the loop. The cultural contribution did not move to the
planner alone: the planner's zone costs are *derived from the CARS reward* (prayer
room 2/3·intensity per step, gender wards 0.1/3, over the 0.8 progress reward of a
full-speed step, scaled ×3), so planner and policy optimise one objective; the
culture-blind vanilla/he2025 policies use the plain shortest path
(`cultural_ratio:=0.0`).

### What `route_planner` does

```
/map (SLAM or map_server) ──┐
zone-label PNG (§6)        ──┼─► inflate by robot radius ─► Dijkstra cost-to-go ─► /rl_goal (x, y, z=route length)
goal (/goal_pose, /nav_goal, /patrol_goal, or goal_x/goal_y) ─┘                  └─► /rl_route (Path, for RViz)
```

`controller_node.py` then builds the goal vector exactly as in training:

```python
b    = atan2(wy - y, wx - x) - yaw            # bearing to the WAYPOINT
dist = z if z > 0 else hypot(gx - x, gy - y)  # remaining ROUTE length
goal = [sin(b), cos(b), min(dist / 25.0, 1.0)]
```

The look-ahead (2.5 m), the line-of-sight pull-back and the cost multipliers are
trained values — change them only with a retrain. `path_to_waypoint.py` (Nav2's
`/plan` → waypoint) remains as a fallback with the same 2.5 m look-ahead and `z`,
but Nav2's planner cannot apply the cultural costs.

### The shield: v2 (swept path) by default

`controller` applies `scan_utils.shield_v2` (parameter `shield_version: 2`) to every
command: it predicts the body's swept path along the commanded steering arc and limits
speed by the distance to the first LiDAR point or tracked person on that path (people
propagated 0.6 s with their velocity; 15 cm margin along the track, 3 cm sideways for
walls, 15 cm sideways for people; a few neighbouring steering angles are tried when the
arc is blocked; reverse is governed the same way). The forward-cone shield (v1, `shield_version: 1`)
could not see side contacts while passing or turning — in the benchmark v2 took every
driver to **zero collisions** (learned policies 94–95 % success, the classical
social-force driver 97.8 %) at the same speed. The shield reads the sector-minimum scan,
so nothing between sampled beams is missed. The same code runs in the simulator
(`experiments/planner/arc_shield.py`); the two implementations are checked equal.

---

## 2. Inventory

### Robot

| Property | Value | Where it comes from |
|---|---|---|
| Platform | Hiwonder MentorPi, Ackermann variant | `MACHINE_TYPE=MentorPi_Acker` |
| Wheelbase *L* | **0.145 m** | `driver/controller/controller/ackermann.py` |
| Track width | 0.133 m | same |
| Wheel diameter | 0.067 m | same |
| Max steering | **45°** (hard clamp) | `ackermann.py`, `set_velocity` |
| Min turning radius | **0.145 m** | *L* / tan δ<sub>max</sub> |
| LiDAR | LDROBOT LD19, 360°, 12 m | `peripherals/launch/include/ldlidar_LD19.launch.py` |
| Compute | Raspberry Pi 5 | — |

### What the policy expects

| Property | Value |
|---|---|
| Observation | 90-dim: 48 LiDAR + 5 robot + 20 pedestrian + 3 goal + 14 context |
| Goal input | waypoint 2.5 m along the planned route + remaining route length (§1) |
| Action | `[v, δ]`, v ∈ [−0.3, 0.8] m/s, δ ∈ [−30°, 30°] |
| Sim wheelbase | 0.20 m |
| Sim min turning radius | 0.35 m |
| Control rate | 10 Hz |

**The simulated envelope is strictly conservative.** The robot turns inside the
sim's tightest arc (0.145 m vs 0.35 m), so every curvature the policy can
command is physically reachable and the servo clamp never engages. No
retraining is required for the chassis difference.

---

## 3. Command translation

The policy and shield emit an Ackermann command; the MentorPi consumes a
velocity Twist. One node bridges them.

```python
# Subscribes: /cmd_shielded   (Twist: linear.x = v m/s, angular.z = steering rad)
# Publishes:  controller/cmd_vel (Twist: linear.x = v m/s, angular.z = yaw rate)

SIM_WHEELBASE = 0.20          # the TRAINING wheelbase -- not the robot's 0.145
omega = (v / SIM_WHEELBASE) * math.tan(delta)
```

### Why the *sim* wheelbase

What the policy intends is a path curvature, κ = tan(δ)/L. Dividing by the sim
wheelbase recovers the curvature it meant. The MentorPi then re-derives its own
steering angle with its own wheelbase, in `odom_publisher_node.cmd_vel_callback`
→ `AckermannChassis.set_velocity`:

```
theta = atan(L_real * omega / v) = atan(0.145 * tan(delta) / 0.20)
      = atan(0.725 * tan(delta))
```

That is exactly the correct curvature-preserving remap between the two chassis,
obtained for free. **Using the real wheelbase in the bridge would over-steer by
~38%.** At the policy's full lock (δ = 30°) the servo sees 22.7°, well inside
its 45° limit.

In reverse (v < 0) both *v* and ω flip sign, so `atan(L·ω/v)` recovers the same
steering angle — correct, since wheels do not counter-steer when backing up.

### Two gotchas that will silently ruin a run

1. **Publish to `controller/cmd_vel`, never plain `cmd_vel`.**
   `odom_publisher_node.app_cmd_vel_callback` clamps the plain topic to
   **±0.2 m/s and ±0.5 rad/s**. The policy trained at 0.8 m/s and needs up to
   0.578 rad/s at full lock, so that path distorts its behaviour without any
   error message. `controller/cmd_vel` reaches `cmd_vel_callback` unclamped.

2. **Do not publish odometry.** The MentorPi stack already publishes `/odom`
   and broadcasts `odom → base_link`. A second publisher fights it.

A `speed_scale` multiplier is useful for early bring-up. Because ω scales with
*v*, the curvature κ = ω/v is unchanged — **scaling speed does not change the
geometric path**, only how fast the robot travels it.

---

## 4. Phase 0 — Prerequisites

On the robot:

```bash
echo $MACHINE_TYPE     # must print MentorPi_Acker
echo $LIDAR_TYPE
ros2 launch bringup bringup.launch.py     # base up, /scan and /odom alive
ros2 topic hz /scan                        # expect ~10 Hz
```

Copy onto the robot, keeping each model with **its matched** normalizer — a
mismatched `vec_normalize.pkl` silently destroys performance:

- `final_model.zip` + `vec_normalize.pkl` for the chosen agent (the campaign's
  headline models are the FINAL checkpoints; `<agent>_final.pt` for cwn/attention)
- the zone-label PNG — used by the controller (context vector) **and** by the
  route planner (cultural route costs)

Paths are set in `ros2_stack/hospitalnav_deploy/config/robot.yaml`
(`controller:` model_zip, vecnorm_pkl, agent, semantic_png, map_w, map_h,
urgency, tod_hour, bearing_offset; `route_planner:` cultural_ratio — 3.0 for
CARS agents, 0.0 for vanilla/he2025 — lookahead, semantic_png, tod_hour).

---

## 5. Phase 1 — Manual drive → SLAM map

```bash
ros2 launch slam slam.launch.py slam_method:=gmapping enable_save:=true
```

Drive the robot manually over the full route, slowly.

- Cover every corridor the robot will later use, in **both** directions.
- Pause at doorways so the scan matcher locks on — doorways are the known hard
  case for these policies and a thin map there costs you later.
- Close at least one loop, returning to the start, so the loop closure corrects
  drift.
- Keep clear of moving people while mapping; they smear the occupancy grid.

Save, then confirm the map exists:

```bash
ls ~/ros2_ws/src/slam/maps/          # <map_name>.yaml + <map_name>.pgm
```

Inspect the `.pgm` before continuing. Doorways must appear as clean gaps, not
partially closed walls — a doorway that maps shut will make every run fail there
and will look like a policy failure when it is a mapping failure.

---

## 6. Phase 2 — Zone labels

The 14-dim context vector needs to know which cultural zone the robot is in
(prayer room, male ward, female ward, corridor, nurse station, …). This comes
from a colour-coded PNG registered to the SLAM map, not from the SLAM map
itself.

1. Copy `<map_name>.pgm` and paint each zone a flat distinct colour.
2. Save alongside the map; point `semantic_png` at it in `robot.yaml`.
3. Set `semantic_res` and `semantic_origin` to **match the map's `.yaml`
   exactly** — resolution and origin must agree or every zone lookup is offset.

Verify before driving: place the robot inside a labelled zone and confirm the
context vector reports that zone.

---

## 7. Phase 3 — Localization

```bash
ros2 launch navigation navigation.launch.py map:=<map_name>
```

Set the initial pose in RViz (2D Pose Estimate), then rotate the robot in place
a little and confirm the scan overlay settles onto the walls. Check the TF chain
is complete:

```bash
ros2 run tf2_tools view_frames      # expect map -> odom -> base_link -> laser
```

Nav2 comes up here for localization (AMCL + map_server). **Its planner is not
used**: the route comes from `route_planner`, which plans on the same `/map`
with the training cost field — see §1.

---

## 8. Phase 4 — Autonomous start-to-goal

Bring-up order, one terminal each:

| # | Node | Purpose |
|---|---|---|
| 1 | `bringup.launch.py` | base, motors, `/scan`, `/odom` |
| 2 | `navigation.launch.py map:=<name>` | localization + costmap |
| 3 | `people_tracker` | populates the 20-dim pedestrian block |
| 4 | `route_planner` | global level: `/map` + zone PNG + goal → `/rl_goal` (waypoint, z = route length), `/rl_route` |
| 5 | `controller` | policy inference + the v2 shield → `/cmd_shielded` (`shield_node` is not needed) |
| 6 | — | (optional classical driver: the tuned social-force controller can replace the policy; same planner, same shield) |
| 7 | *bridge* | `/cmd_shielded` → `controller/cmd_vel` (§3) |
| 8 | *goal* | RViz "2D Goal Pose" (`/goal_pose`), or `goal_publisher -p topic:=/nav_goal` |
| 9 | `csv_logger` | records the run |

`hospitalnav_bringup.launch.py` starts 2–5 (+ localization); set the destination
in map coordinates either by clicking in RViz or by parameter:

```bash
ros2 run hospitalnav_deploy goal_publisher --ros-args \
    -p topic:=/nav_goal -p goal_x:=<x> -p goal_y:=<y>
```

Check in RViz that `/rl_route` runs through doorways and (for a CARS agent)
around the prayer room before the robot moves: a route through a wall means the
planner's map/zone registration is wrong (§6), not the policy.

### Before the first floor run

- **Wheels off the ground.** Confirm the robot drives forward on a forward
  command, and that a positive steering command turns the wheels the expected
  way. A sign error here is the single most common bring-up fault.
- **Watchdog.** Kill the controller mid-command; the robot must stop, not
  coast.
- **Speed.** Start with `speed_scale:=0.4`. The path geometry is unchanged, so
  this is a free safety margin.
- **E-stop reachable.** 0.8 m/s is quick for an indoor robot near people.

---

## 9. Recording results

`csv_logger` writes `t, v_cmd, steering_cmd, x, y, yaw, goal_x, goal_y,
goal_dist`. Keep, per run: the agent and seed, the checkpoint's sha256, the
start and goal coordinates, and the map used.

### Reporting convention — important

Report physical runs as **per-policy binary outcomes**: "reached" or "not
reached", with the failure mode where relevant (halted at doorway, collision,
off-route, timeout). Repeated attempts may be reported as "reached in N of M
attempts."

**Never synthesize a success rate from bring-up runs.** Single deterministic
demonstrations are not comparable to the 500-episode benchmark success rates —
different denominators, different experiment. Quoting a rate invites exactly the
false comparison a reviewer would flag as a category error.

---

## 10. Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Nothing moves, motors silent | `rrc_bridge.py` in the loop — it targets Yahboom's `Rosmaster_Lib`, which this robot does not have | Use the §3 bridge; retire `rrc_bridge.py` |
| Moves, but sluggish and under-turns | Publishing to plain `cmd_vel` (clamped to 0.2 m/s / 0.5 rad/s) | Publish to `controller/cmd_vel` |
| Over-steers by roughly a third | Bridge used *L* = 0.145 instead of the sim's 0.20 | Divide by `SIM_WHEELBASE` |
| Wanders, ignores the goal, oscillates | `/rl_goal` without `z` (goal_publisher on `/rl_goal`, old bridge): goal channel reads the 2.5 m waypoint distance (0.1) | Run `route_planner`; publish the destination to `/nav_goal` (§1, §8) |
| Drives into a wall or a dead end | Route planned on a wrong grid: `/map` origin/resolution vs zone PNG, or unknown cells treated as free | Check `/rl_route` in RViz; fix the registration (§6); keep `unknown_is_free: false` |
| Mapless (`*_mapless`) policy with the planner, or a hierarchical policy with the final goal | train/deploy wiring mismatch | §1 table |
| Ignores cultural zones | `semantic_png` missing, or its origin/resolution disagree with the map | Re-register the label map (§6) |
| Good in one spot, bad elsewhere | Localization drift | Re-run Phase 1 with a proper loop closure |
| Consistently stuck at one doorway | Doorway mapped partially closed | Inspect the `.pgm`; re-map that area |
| Erratic from the first command | `vec_normalize.pkl` not the one matched to this checkpoint | Re-copy the matched pair |

---

## 11. Scope — what this does and does not establish

This procedure produces **per-policy demonstrations on a real robot in a real
building**. It does not produce a quantitative transfer rate, and it is not
comparable to the benchmark tables.

The honest claim it supports: unchanged benchmark checkpoints, carried onto a
SLAM-built map of a previously unseen environment, drive a physical Ackermann
robot from start to goal without retraining. That is a sim-to-real transfer
demonstration — which is what the paper's limitations section already states is
the remaining step, and it claims nothing more.
