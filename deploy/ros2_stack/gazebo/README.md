# Gazebo validation — run the trained policy, no retraining

Validate the exact same trained policy + shield in a 3-D physics sim before the
real robot. **You do not retrain** — Gazebo just replaces the real sensors as a
source of `/scan`, `/odom`, and (optionally) `/people`; the same
`hospitalnav_deploy` controller + shield run unchanged.

**Read [`ENVIRONMENT_SPEC.md`](ENVIRONMENT_SPEC.md) first** — it is the contract of
numbers (map 40x30, LiDAR 360/12 m, Ackermann L=0.20 m, the 90-D obs) the Gazebo
world/robot must match so the policy transfers.

## What's here
```
ENVIRONMENT_SPEC.md          the "no-retraining" contract
gen_gazebo_world.py          builds worlds/hospital.world from configs/dmch_hospital.yaml
urdf/hospitalbot.urdf        Ackermann robot + 360/12 m LiDAR (matches the sim)
launch/gazebo_sim.launch.py  gz sim + robot + bridges + policy + shield
```
The two Gazebo-specific ROS nodes live in the package:
`hospitalnav_deploy/gz_cmd_adapter.py` (steering->yaw-rate) and
`gt_pedestrians.py` (ground-truth crowd -> /people).

## Prereqs (on WSL/Ubuntu 22.04, ROS 2 Humble)
```bash
sudo apt install -y ros-humble-ros-gz ros-humble-ros-gz-bridge ros-humble-ros-gz-sim
```

## Run
```bash
# 1) generate the world (matches the trained env exactly)
cd hardware_deploy/gazebo && python3 gen_gazebo_world.py
# 2) build the package (once) and source
cd ~/hn_ws && colcon build && source install/setup.bash
# 3) launch: gz + robot + policy + shield  (put your model in ~/models)
ros2 launch $PWD/../hardware_deploy/gazebo/launch/gazebo_sim.launch.py \
     model_zip:=$HOME/models/best_model.zip vecnorm_pkl:=$HOME/models/vec_normalize.pkl \
     agent:=cwn weight_pt:=$HOME/models/cwn_best.pt goal_x:=8.0 goal_y:=4.0
```
The robot spawns at the nurse station and drives toward the goal (default: the
emergency dept) using the policy, with the shield preventing collisions. Watch it
in the Gazebo GUI and/or RViz; `ros2 topic echo /cmd_shielded` and
`/policy_weights` show the decisions.

## Adding the crowd (optional)
1. Add `<actor>` elements (walking people) to `worlds/hospital.world`.
2. Bridge their world poses to `/pedestrian_poses` (a `geometry_msgs/PoseArray`,
   map frame) via `ros_gz_bridge`.
3. Run `ros2 run hospitalnav_deploy gt_pedestrians`.
Now the policy reacts to real moving people (ground-truth, no detector needed).
Without this, `/people` stays zeros = "no one nearby" (same as the sim's padding).

## SLAM + Nav2 in Gazebo (optional, richer)
Everything from the main `DEPLOYMENT_GUIDE.md` (slam_toolbox, Nav2,
path_to_waypoint) works in Gazebo too - point them at Gazebo's `/scan` and
`/odom` instead of a fixed goal. Gazebo is a safe place to get that whole stack
working before the real robot.

## Expected gap (do NOT retrain for it)
Gazebo has real physics (inertia, friction) vs the idealized bicycle model, and
scripted actors vs the sim's HSFM crowd. The reactive policy + shield tolerate
this; expect a modest performance drop. Tune the URDF inertias/friction to reduce
slip if needed. Retraining IN Gazebo (with domain randomization) is only for
maximal real transfer - a separate, optional effort.
