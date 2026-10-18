# Gazebo (gz-sim Harmonic) campaign - real MentorPi chassis, trained runs_v5 policies

Every row is ONE run of the nurse-desk (37, 15) -> emergency-room (5, 7.5) task with the
trained route planner choosing the path on the prebuilt map; `reached` = within 0.4 m
(training GOAL_TOL) inside the 300 s simulation-time budget. Localisation is ground truth
(gz OdometryPublisher), the lidar is the LD19-spec gpu_lidar the LiDAR people tracker
reads. Binary outcomes only - no success rate is synthesised from these demonstration runs.

PEDESTRIANS. `busy` = 8 designed walkers, `crowd45` = those 8 + 37 generated ones
(sim/gen_crowd_routes.py: loops planned on the walled map, every room entered through its
doorway). Each walks its own loop under the social force model (Helbing & Molnar 1995;
hospitalnav_gz_plugins PedestrianBodies): people keep their route but give each other and
the robot room - slowing, stopping, stepping around - and never enter a wall. Each has a
70 kg collision body, so contact is physical: the robot cannot drive through a person and a
person shoves the robot, with a bounded force. Ground truth (contacts, distances) is the
simulated walker state, logged per run to peds.csv. `max tilt` checks the robot stayed
upright (> 45 deg would be a knock-down); it is blank for runs logged before the column existed.

Prayer-zone depth is the deepest point reached measured from the zone's nearest edge:
the cultural planner routes along the zone's edge (open tier), so a cultural policy
skims it by a few decimetres while a culture-blind one crosses the room (~4 m deep).
Superseded runs are in `results_superseded/` with the reason (2026-10-16 ground-truth
timing; 2026-10-16 ghost pedestrians; 2026-10-17 stiff pedestrian bodies that could fling or
topple the robot) and are not reported here.

## walled tier, pedestrians: crowd45

| policy | escape on | escape off |
|---|---|---|
| residual_sf | reached | timeout |
| classical_sf2 | timeout | timeout |
| fixed_equal | reached | reached |
| vanilla | reached | reached |
| risk_field | reached | reached |
| mapless_fixed_equal | timeout | timeout |

| tier | policy | peds | escape | outcome | time to goal (sim s) | path (m) | escapes stall/wedge | prayer zone s / max depth m | gender wards s | ped min (m) | ped contacts (<0.5 m) | touches (<0.35 m) | personal zone (s) | max tilt (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walled | residual_sf | crowd45 | on | **reached** | 202.2 | 47.34 | 3/0 | 0.0 / 0.00 | 0.0 | 0.34 | 7 | 1 | 173.6 | 0.8 |
| walled | residual_sf | crowd45 | off | **timeout** | - | 17.92 | 0/0 | 0.0 / 0.00 | 0.0 | 0.34 | 3 | 1 | 283.4 | 0.8 |
| walled | classical_sf2 | crowd45 | on | **timeout** | - | 33.81 | 11/0 | 0.0 / 0.00 | 0.0 | 0.34 | 22 | 3 | 281.8 | 0.8 |
| walled | classical_sf2 | crowd45 | off | **timeout** | - | 9.97 | 0/0 | 0.0 / 0.00 | 0.0 | 0.36 | 2 | 0 | 286.0 | 0.7 |
| walled | fixed_equal | crowd45 | on | **reached** | 134.6 | 41.17 | 0/0 | 0.0 / 0.00 | 0.0 | 0.41 | 8 | 0 | 109.0 | 0.8 |
| walled | fixed_equal | crowd45 | off | **reached** | 221.6 | 39.76 | 0/0 | 0.0 / 0.00 | 0.0 | 0.32 | 11 | 4 | 196.4 | 0.9 |
| walled | vanilla | crowd45 | on | **reached** | 130.6 | 40.53 | 0/0 | 0.0 / 0.00 | 0.0 | 0.34 | 11 | 1 | 104.6 | 0.8 |
| walled | vanilla | crowd45 | off | **reached** | 138.0 | 40.63 | 0/0 | 0.0 / 0.00 | 0.0 | 0.38 | 12 | 0 | 105.6 | 0.8 |
| walled | risk_field | crowd45 | on | **reached** | 121.4 | 48.19 | 0/0 | 0.0 / 0.00 | 0.0 | 0.44 | 9 | 0 | 81.4 | 0.8 |
| walled | risk_field | crowd45 | off | **reached** | 136.8 | 41.26 | 0/0 | 0.0 / 0.00 | 0.0 | 0.43 | 9 | 0 | 108.6 | 0.9 |
| walled | mapless_fixed_equal | crowd45 | on | **timeout** | - | 30.49 | 6/0 | 0.0 / 0.00 | 0.0 | 0.33 | 10 | 1 | 235.0 | 0.9 |
| walled | mapless_fixed_equal | crowd45 | off | **timeout** | - | 29.63 | 0/0 | 0.0 / 0.00 | 0.0 | 0.41 | 1 | 0 | 276.0 | 0.7 |

## open tier, pedestrians: crowd45

| policy | escape on | escape off |
|---|---|---|
| residual_sf | reached | reached |
| classical_sf2 | reached | reached |
| fixed_equal | reached | reached |
| vanilla | reached | reached |
| risk_field | reached | timeout |
| mapless_fixed_equal | reached | timeout |

| tier | policy | peds | escape | outcome | time to goal (sim s) | path (m) | escapes stall/wedge | prayer zone s / max depth m | gender wards s | ped min (m) | ped contacts (<0.5 m) | touches (<0.35 m) | personal zone (s) | max tilt (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| open | residual_sf | crowd45 | on | **reached** | 70.8 | 39.50 | 0/0 | 13.3 / 0.59 | 0.0 | 0.57 | 0 | 0 | 23.0 | 0.7 |
| open | residual_sf | crowd45 | off | **reached** | 70.8 | 38.23 | 0/0 | 14.1 / 0.54 | 0.0 | 0.55 | 0 | 0 | 25.0 | 0.7 |
| open | classical_sf2 | crowd45 | on | **reached** | 70.0 | 36.93 | 0/0 | 12.1 / 0.46 | 0.0 | 0.54 | 0 | 0 | 24.6 | 0.8 |
| open | classical_sf2 | crowd45 | off | **reached** | 72.0 | 37.96 | 0/0 | 12.8 / 0.51 | 0.0 | 0.55 | 0 | 0 | 26.0 | 0.6 |
| open | fixed_equal | crowd45 | on | **reached** | 81.0 | 37.49 | 0/0 | 6.0 / 0.30 | 0.0 | 0.46 | 3 | 0 | 44.4 | 0.8 |
| open | fixed_equal | crowd45 | off | **reached** | 96.0 | 38.75 | 0/0 | 14.6 / 0.78 | 0.0 | 0.45 | 1 | 0 | 48.0 | 0.8 |
| open | vanilla | crowd45 | on | **reached** | 81.2 | 37.93 | 1/0 | 13.9 / 4.00 | 0.0 | 0.58 | 0 | 0 | 8.0 | 0.6 |
| open | vanilla | crowd45 | off | **reached** | 68.0 | 37.38 | 0/0 | 13.9 / 3.98 | 0.0 | 0.91 | 0 | 0 | 6.2 | 0.7 |
| open | risk_field | crowd45 | on | **reached** | 70.6 | 39.38 | 0/0 | 14.6 / 3.99 | 0.0 | 0.79 | 0 | 0 | 6.2 | 0.8 |
| open | risk_field | crowd45 | off | **timeout** | - | 5.79 | 0/0 | 0.0 / 0.00 | 0.0 | 0.87 | 0 | 0 | 0.8 | 0.9 |
| open | mapless_fixed_equal | crowd45 | on | **reached** | 231.8 | 50.81 | 11/0 | 14.3 / 3.88 | 0.0 | 0.64 | 0 | 0 | 23.2 | 0.9 |
| open | mapless_fixed_equal | crowd45 | off | **timeout** | - | 20.68 | 0/0 | 0.0 / 0.00 | 0.0 | 0.78 | 0 | 0 | 3.2 | 0.6 |

## walled tier, pedestrians: busy

| policy | escape on | escape off |
|---|---|---|
| residual_sf | reached | timeout |
| classical_sf2 | reached | timeout |
| fixed_equal | reached | reached |
| vanilla | reached | timeout |
| risk_field | reached | reached |
| mapless_fixed_equal | reached | timeout |

| tier | policy | peds | escape | outcome | time to goal (sim s) | path (m) | escapes stall/wedge | prayer zone s / max depth m | gender wards s | ped min (m) | ped contacts (<0.5 m) | touches (<0.35 m) | personal zone (s) | max tilt (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walled | residual_sf | busy | on | **reached** | 195.0 | 62.28 | 8/0 | 0.0 / 0.00 | 0.0 | 0.52 | 0 | 0 | 31.8 | 0.8 |
| walled | residual_sf | busy | off | **timeout** | - | 43.03 | 0/0 | 0.0 / 0.00 | 0.0 | 0.52 | 0 | 0 | 49.6 | 0.8 |
| walled | classical_sf2 | busy | on | **reached** | 138.0 | 57.49 | 1/0 | 0.0 / 0.00 | 0.0 | 0.45 | 2 | 0 | 30.2 | 0.8 |
| walled | classical_sf2 | busy | off | **timeout** | - | 23.10 | 0/0 | 0.0 / 0.00 | 0.0 | 0.33 | 7 | 3 | 82.4 | 0.7 |
| walled | fixed_equal | busy | on | **reached** | 75.6 | 41.38 | 0/0 | 0.0 / 0.00 | 0.0 | 0.49 | 1 | 0 | 15.2 | 0.8 |
| walled | fixed_equal | busy | off | **reached** | 75.2 | 40.56 | 0/0 | 0.0 / 0.00 | 0.0 | 0.44 | 1 | 0 | 13.4 | 0.7 |
| walled | vanilla | busy | on | **reached** | 75.4 | 41.37 | 0/0 | 0.0 / 0.00 | 0.0 | 0.49 | 1 | 0 | 12.8 | 0.9 |
| walled | vanilla | busy | off | **timeout** | - | 4.99 | 0/0 | 0.0 / 0.00 | 0.0 | 0.84 | 0 | 0 | 16.4 | 0.6 |
| walled | risk_field | busy | on | **reached** | 75.4 | 41.51 | 0/0 | 0.0 / 0.00 | 0.0 | 0.40 | 2 | 0 | 13.0 | 0.7 |
| walled | risk_field | busy | off | **reached** | 80.0 | 42.81 | 0/0 | 0.0 / 0.00 | 0.0 | 0.38 | 1 | 0 | 13.4 | 0.8 |
| walled | mapless_fixed_equal | busy | on | **reached** | 131.0 | 46.95 | 4/0 | 0.0 / 0.00 | 0.0 | 0.45 | 1 | 0 | 23.2 | 0.8 |
| walled | mapless_fixed_equal | busy | off | **timeout** | - | 16.75 | 0/0 | 0.0 / 0.00 | 0.0 | 0.44 | 1 | 0 | 42.8 | 0.7 |

## open tier, pedestrians: busy

| policy | escape on | escape off |
|---|---|---|
| residual_sf | reached | reached |
| classical_sf2 | reached | reached |
| fixed_equal | reached | reached |
| vanilla | reached | timeout |
| risk_field | reached | timeout |
| mapless_fixed_equal | reached | reached |

| tier | policy | peds | escape | outcome | time to goal (sim s) | path (m) | escapes stall/wedge | prayer zone s / max depth m | gender wards s | ped min (m) | ped contacts (<0.5 m) | touches (<0.35 m) | personal zone (s) | max tilt (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| open | residual_sf | busy | on | **reached** | 67.2 | 38.59 | 0/0 | 11.8 / 0.66 | 0.0 | 0.84 | 0 | 0 | 6.0 | 0.7 |
| open | residual_sf | busy | off | **reached** | 68.0 | 38.79 | 0/0 | 11.3 / 0.54 | 0.0 | 0.78 | 0 | 0 | 5.8 | 0.8 |
| open | classical_sf2 | busy | on | **reached** | 62.6 | 35.97 | 0/0 | 6.0 / 0.40 | 0.0 | 0.98 | 0 | 0 | 4.0 | 0.8 |
| open | classical_sf2 | busy | off | **reached** | 62.2 | 35.81 | 0/0 | 5.5 / 0.39 | 0.0 | 0.98 | 0 | 0 | 3.8 | 0.4 |
| open | fixed_equal | busy | on | **reached** | 69.0 | 37.17 | 0/0 | 12.5 / 0.44 | 0.0 | 0.55 | 0 | 0 | 10.8 | 0.7 |
| open | fixed_equal | busy | off | **reached** | 68.8 | 37.07 | 0/0 | 9.1 / 0.62 | 0.0 | 0.60 | 0 | 0 | 12.6 | 0.8 |
| open | vanilla | busy | on | **reached** | 67.6 | 36.92 | 0/0 | 13.9 / 3.98 | 0.0 | 0.57 | 0 | 0 | 3.0 | 0.7 |
| open | vanilla | busy | off | **timeout** | - | 5.07 | 0/0 | 0.0 / 0.00 | 0.0 | 0.90 | 0 | 0 | 1.4 | 0.8 |
| open | risk_field | busy | on | **reached** | 111.4 | 42.25 | 2/0 | 13.9 / 3.99 | 0.0 | 0.78 | 0 | 0 | 1.8 | 0.8 |
| open | risk_field | busy | off | **timeout** | - | 19.49 | 0/0 | 0.0 / 0.00 | 0.0 | 0.81 | 0 | 0 | 1.8 | 0.7 |
| open | mapless_fixed_equal | busy | on | **reached** | 133.6 | 44.78 | 5/0 | 13.9 / 1.59 | 0.0 | 0.48 | 1 | 0 | 13.4 | 0.9 |
| open | mapless_fixed_equal | busy | off | **reached** | 66.2 | 38.33 | 0/0 | 14.7 / 2.02 | 0.0 | 0.79 | 0 | 0 | 5.2 | 0.7 |

## walled tier, pedestrians: none

| policy | escape on | escape off |
|---|---|---|
| residual_sf | reached | - |
| classical_sf2 | timeout | - |
| fixed_equal | reached | - |
| vanilla | reached | - |
| risk_field | reached | - |
| mapless_fixed_equal | reached | - |

| tier | policy | peds | escape | outcome | time to goal (sim s) | path (m) | escapes stall/wedge | prayer zone s / max depth m | gender wards s | ped min (m) | ped contacts (<0.5 m) | touches (<0.35 m) | personal zone (s) | max tilt (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walled | mapless_fixed_equal | none | on | **reached** | 95.2 | 44.39 | 2/0 | 0.0 / 0.00 | 0.0 | - | 0 | 0 | 0.0 | - |
| walled | classical_sf2 | none | on | **timeout** | - | 51.05 | 18/0 | 0.0 / 0.00 | 0.0 | - | 0 | 0 | 0.0 | - |
| walled | fixed_equal | none | on | **reached** | 82.2 | 41.96 | 1/0 | 0.0 / 0.00 | 0.0 | - | 0 | 0 | 0.0 | - |
| walled | residual_sf | none | on | **reached** | 199.4 | 61.81 | 9/0 | 0.0 / 0.00 | 0.0 | - | 0 | 0 | 0.0 | - |
| walled | risk_field | none | on | **reached** | 84.8 | 43.64 | 1/0 | 0.0 / 0.00 | 0.0 | - | 0 | 0 | 0.0 | - |
| walled | vanilla | none | on | **reached** | 84.4 | 43.23 | 1/0 | 0.0 / 0.00 | 0.0 | - | 0 | 0 | 0.0 | - |

## open tier, pedestrians: none

| policy | escape on | escape off |
|---|---|---|
| residual_sf | reached | - |
| classical_sf2 | reached | - |
| fixed_equal | reached | - |
| vanilla | reached | - |
| risk_field | reached | - |
| mapless_fixed_equal | reached | - |

| tier | policy | peds | escape | outcome | time to goal (sim s) | path (m) | escapes stall/wedge | prayer zone s / max depth m | gender wards s | ped min (m) | ped contacts (<0.5 m) | touches (<0.35 m) | personal zone (s) | max tilt (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| open | mapless_fixed_equal | none | on | **reached** | 64.6 | 37.79 | 0/0 | 14.1 / 2.50 | 0.0 | - | 0 | 0 | 0.0 | - |
| open | classical_sf2 | none | on | **reached** | 62.0 | 36.06 | 0/0 | 2.7 / 0.38 | 0.0 | - | 0 | 0 | 0.0 | - |
| open | fixed_equal | none | on | **reached** | 65.6 | 36.69 | 0/0 | 11.0 / 0.74 | 0.0 | - | 0 | 0 | 0.0 | - |
| open | residual_sf | none | on | **reached** | 66.6 | 38.47 | 0/0 | 13.6 / 0.61 | 0.0 | - | 0 | 0 | 0.0 | - |
| open | risk_field | none | on | **reached** | 75.2 | 38.41 | 1/0 | 13.8 / 4.00 | 0.0 | - | 0 | 0 | 0.0 | - |
| open | vanilla | none | on | **reached** | 77.2 | 38.39 | 0/0 | 13.6 / 3.99 | 0.0 | - | 0 | 0 | 0.0 | - |

## Realism tiers (photogrammetry assets; furnished = unmapped clutter)

| tier | policy | peds | escape | outcome | time to goal (sim s) | path (m) | escapes stall/wedge | prayer zone s / max depth m | gender wards s | ped min (m) | ped contacts (<0.5 m) | touches (<0.35 m) | personal zone (s) | max tilt (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | residual_sf | busy | on | **reached** | 149.2 | 56.20 | 4/0 | 0.0 / 0.00 | 0.0 | 0.49 | 1 | 0 | 35.8 | 0.8 |
| real | classical_sf2 | busy | on | **reached** | 201.4 | 57.58 | 7/0 | 0.0 / 0.00 | 0.0 | 0.38 | 6 | 0 | 54.2 | 0.8 |
| real | fixed_equal | busy | on | **reached** | 84.0 | 41.61 | 1/0 | 0.0 / 0.00 | 0.0 | 0.39 | 1 | 0 | 17.4 | 0.7 |
| real | vanilla | busy | on | **reached** | 122.4 | 43.69 | 2/0 | 0.0 / 0.00 | 0.0 | 0.49 | 1 | 0 | 21.6 | 0.8 |
| furnished | residual_sf | busy | on | **reached** | 140.4 | 54.82 | 4/0 | 0.0 / 0.00 | 0.0 | 0.48 | 1 | 0 | 35.4 | 0.8 |
| furnished | classical_sf2 | busy | on | **timeout** | - | 62.45 | 14/0 | 0.0 / 0.00 | 0.0 | 0.38 | 4 | 0 | 48.6 | 0.9 |
| furnished | fixed_equal | busy | on | **reached** | 86.8 | 41.81 | 1/0 | 0.0 / 0.00 | 0.0 | 0.39 | 2 | 0 | 13.6 | 0.8 |
| furnished | vanilla | busy | on | **reached** | 87.2 | 41.80 | 1/0 | 0.0 / 0.00 | 0.0 | 0.34 | 1 | 1 | 14.8 | 0.7 |

### Policies

- `residual_sf`: residual RL over the social-force prior (recommended)
- `classical_sf2`: tuned social force, no model
- `fixed_equal`: pure learned CARS policy
- `vanilla`: culture-blind contrast (geo planner)
- `risk_field`: risk-field baseline (He 2025 style)
- `mapless_fixed_equal`: no planner - straight-line goal

## Scan invariance (what the policy sees vs the trained geometry)

- `walled`: PASS - policy 48-ray input differs by at most 0.14 mm over 46 poses (raw rays: max 0.000 m at 0 poses, grazing long-range wall hits)
- `real`: PASS - policy 48-ray input differs by at most 1.64 mm over 46 poses (raw rays: max 0.049 m at 17 poses, grazing long-range wall hits)
- `open`: PASS - policy 48-ray input differs by at most 0.21 mm over 46 poses (raw rays: max 0.000 m at 0 poses, grazing long-range wall hits)

## Figures

- `results_figures/gazebo_furnished_chase.png`
- `results_figures/gazebo_furnished_corridor.png`
- `results_figures/gazebo_furnished_emergency.png`
- `results_figures/gazebo_furnished_floorplan.png`
- `results_figures/gazebo_furnished_nursedesk.png`
- `results_figures/gazebo_furnished_nursedesk_east.png`
- `results_figures/gazebo_furnished_oblique.png`
- `results_figures/gazebo_furnished_prayer.png`
- `results_figures/gazebo_furnished_ward.png`
- `results_figures/gazebo_open_chase.png`
- `results_figures/gazebo_open_corridor.png`
- `results_figures/gazebo_open_emergency.png`
- `results_figures/gazebo_open_floorplan.png`
- `results_figures/gazebo_open_nursedesk.png`
- `results_figures/gazebo_open_nursedesk_east.png`
- `results_figures/gazebo_open_oblique.png`
- `results_figures/gazebo_open_prayer.png`
- `results_figures/gazebo_open_ward.png`
- `results_figures/gazebo_real_chase.png`
- `results_figures/gazebo_real_corridor.png`
- `results_figures/gazebo_real_emergency.png`
- `results_figures/gazebo_real_floorplan.png`
- `results_figures/gazebo_real_nursedesk.png`
- `results_figures/gazebo_real_nursedesk_east.png`
- `results_figures/gazebo_real_oblique.png`
- `results_figures/gazebo_real_prayer.png`
- `results_figures/gazebo_real_ward.png`
- `results_figures/gazebo_walled_chase.png`
- `results_figures/gazebo_walled_corridor.png`
- `results_figures/gazebo_walled_emergency.png`
- `results_figures/gazebo_walled_floorplan.png`
- `results_figures/gazebo_walled_nursedesk.png`
- `results_figures/gazebo_walled_nursedesk_east.png`
- `results_figures/gazebo_walled_oblique.png`
- `results_figures/gazebo_walled_prayer.png`
- `results_figures/gazebo_walled_ward.png`
- `results_figures/path_furnished__classical_sf2__escape-on__peds-busy.png`
- `results_figures/path_furnished__fixed_equal__escape-on__peds-busy.png`
- `results_figures/path_furnished__residual_sf__escape-on__peds-busy.png`
- `results_figures/path_furnished__vanilla__escape-on__peds-busy.png`
- `results_figures/path_open__classical_sf2__escape-off__peds-busy.png`
- `results_figures/path_open__classical_sf2__escape-off__peds-crowd45.png`
- `results_figures/path_open__classical_sf2__escape-on__peds-busy.png`
- `results_figures/path_open__classical_sf2__escape-on__peds-crowd45.png`
- `results_figures/path_open__classical_sf2__escape-on__peds-none.png`
- `results_figures/path_open__fixed_equal__escape-off__peds-busy.png`
- `results_figures/path_open__fixed_equal__escape-off__peds-crowd45.png`
- `results_figures/path_open__fixed_equal__escape-on__peds-busy.png`
- `results_figures/path_open__fixed_equal__escape-on__peds-crowd45.png`
- `results_figures/path_open__fixed_equal__escape-on__peds-none.png`
- `results_figures/path_open__mapless_fixed_equal__escape-off__peds-busy.png`
- `results_figures/path_open__mapless_fixed_equal__escape-off__peds-crowd45.png`
- `results_figures/path_open__mapless_fixed_equal__escape-on__peds-busy.png`
- `results_figures/path_open__mapless_fixed_equal__escape-on__peds-crowd45.png`
- `results_figures/path_open__mapless_fixed_equal__escape-on__peds-none.png`
- `results_figures/path_open__residual_sf__escape-off__peds-busy.png`
- `results_figures/path_open__residual_sf__escape-off__peds-crowd45.png`
- `results_figures/path_open__residual_sf__escape-on__peds-busy.png`
- `results_figures/path_open__residual_sf__escape-on__peds-crowd45.png`
- `results_figures/path_open__residual_sf__escape-on__peds-none.png`
- `results_figures/path_open__risk_field__escape-off__peds-busy.png`
- `results_figures/path_open__risk_field__escape-off__peds-crowd45.png`
- `results_figures/path_open__risk_field__escape-on__peds-busy.png`
- `results_figures/path_open__risk_field__escape-on__peds-crowd45.png`
- `results_figures/path_open__risk_field__escape-on__peds-none.png`
- `results_figures/path_open__vanilla__escape-off__peds-busy.png`
- `results_figures/path_open__vanilla__escape-off__peds-crowd45.png`
- `results_figures/path_open__vanilla__escape-on__peds-busy.png`
- `results_figures/path_open__vanilla__escape-on__peds-crowd45.png`
- `results_figures/path_open__vanilla__escape-on__peds-none.png`
- `results_figures/path_real__classical_sf2__escape-on__peds-busy.png`
- `results_figures/path_real__fixed_equal__escape-on__peds-busy.png`
- `results_figures/path_real__residual_sf__escape-on__peds-busy.png`
- `results_figures/path_real__vanilla__escape-on__peds-busy.png`
- `results_figures/path_walled__classical_sf2__escape-off__peds-busy.png`
- `results_figures/path_walled__classical_sf2__escape-off__peds-crowd45.png`
- `results_figures/path_walled__classical_sf2__escape-on__peds-busy.png`
- `results_figures/path_walled__classical_sf2__escape-on__peds-crowd45.png`
- `results_figures/path_walled__classical_sf2__escape-on__peds-none.png`
- `results_figures/path_walled__fixed_equal__escape-off__peds-busy.png`
- `results_figures/path_walled__fixed_equal__escape-off__peds-crowd45.png`
- `results_figures/path_walled__fixed_equal__escape-on__peds-busy.png`
- `results_figures/path_walled__fixed_equal__escape-on__peds-crowd45.png`
- `results_figures/path_walled__fixed_equal__escape-on__peds-none.png`
- `results_figures/path_walled__mapless_fixed_equal__escape-off__peds-busy.png`
- `results_figures/path_walled__mapless_fixed_equal__escape-off__peds-crowd45.png`
- `results_figures/path_walled__mapless_fixed_equal__escape-on__peds-busy.png`
- `results_figures/path_walled__mapless_fixed_equal__escape-on__peds-crowd45.png`
- `results_figures/path_walled__mapless_fixed_equal__escape-on__peds-none.png`
- `results_figures/path_walled__residual_sf__escape-off__peds-busy.png`
- `results_figures/path_walled__residual_sf__escape-off__peds-crowd45.png`
- `results_figures/path_walled__residual_sf__escape-on__peds-busy.png`
- `results_figures/path_walled__residual_sf__escape-on__peds-crowd45.png`
- `results_figures/path_walled__residual_sf__escape-on__peds-none.png`
- `results_figures/path_walled__risk_field__escape-off__peds-busy.png`
- `results_figures/path_walled__risk_field__escape-off__peds-crowd45.png`
- `results_figures/path_walled__risk_field__escape-on__peds-busy.png`
- `results_figures/path_walled__risk_field__escape-on__peds-crowd45.png`
- `results_figures/path_walled__risk_field__escape-on__peds-none.png`
- `results_figures/path_walled__vanilla__escape-off__peds-busy.png`
- `results_figures/path_walled__vanilla__escape-off__peds-crowd45.png`
- `results_figures/path_walled__vanilla__escape-on__peds-busy.png`
- `results_figures/path_walled__vanilla__escape-on__peds-crowd45.png`
- `results_figures/path_walled__vanilla__escape-on__peds-none.png`
- `results_figures/paths_open_peds-busy_compare.png`
- `results_figures/paths_open_peds-crowd45_compare.png`
- `results_figures/paths_open_peds-none_compare.png`
- `results_figures/paths_walled_peds-busy_compare.png`
- `results_figures/paths_walled_peds-crowd45_compare.png`
- `results_figures/paths_walled_peds-none_compare.png`
