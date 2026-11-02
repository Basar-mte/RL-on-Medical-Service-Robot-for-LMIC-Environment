# Feasibility trials on a mobile robot (30 October 2026)

Nine single trials of the simulation-trained residual policy, with unchanged weights, on a
Hiwonder MentorPi, an Ackermann-steering mobile robot that stood in for a service robot because we had no hospital service robot. The site is an apartment with hospital roles
assigned to its rooms (nurse station start, emergency-room goal behind a 0.75 to 0.8 m gate).
Planned route 10.53 m, speed cap 0.45 m/s, arrival within 0.35 m of the goal.

- `real_robot_trials.csv`: one row per trial (outcome, time, path, route deviation, escapes,
  final distance and how it was measured, configuration C0 to C4).
- `real_robot_timeseries.csv`: the 5 Hz run logs (localised pose, commands, filter output,
  remaining route, stall flag, battery).
- `DATA_README.md`: column definitions.

Scope: single runs on one day and one route, with a configuration that changed between trials.
No zone map was loaded and pedestrians were not tracked, so these data support feasibility only.
They give no success rate, no prayer-room result and no contact or distance measure.
