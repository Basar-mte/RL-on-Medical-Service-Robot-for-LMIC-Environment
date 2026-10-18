# LMIC-HospitalNav-v1 — Hardware Deployment (single folder)

Everything to run the trained policy + reactive safety shield on a real Ackermann
robot (Yahboom A1 chassis, 360°/12 m LDROBOT LD19 LiDAR, Raspberry Pi 5, RRC-Lite)
with **SLAM mapping** and **Nav2 navigation**. **This whole folder is what you copy
to the Pi** and `colcon build`.

**New to ROS 2 / robots? Read [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)** (or the
Word version `HospitalNav_Hardware_Deployment.docx`) — a step-by-step beginner
guide: WSL Ubuntu 22.04 (**ROS 2 Humble**) → Raspberry Pi 5, how to feed the Pi,
how to monitor, and what the results look like. Every node's code is inline in it.

## What's here
```
setup_pi5.sh                 install ROS 2 Humble + SLAM/Nav2 + deps (WSL and Pi)
requirements-pi.txt          python deps
DEPLOYMENT_GUIDE.md / .docx  the full beginner guide (all code inline)
docs/real_world_test_protocol.md   how to run + report the real-world demo (for reviewers)
hospitalnav_deploy/          the ROS 2 package — colcon build this
  hospitalnav_deploy/        nodes: lidar_node, sim_stub, controller_node, shield_node,
                             perception, path_to_waypoint, rrc_bridge, scan_utils, ld19_driver
  config/                    nav2_hospitalnav.yaml, robot.yaml
  launch/hospitalnav_bringup.launch.py
  test/test_obs_shapes.py    runs with no ROS / no hardware
```

## 60-second dry-run (no robot)
```bash
bash setup_pi5.sh                                  # once (WSL Ubuntu 22.04)
mkdir -p ~/hn_ws/src && cp -r hospitalnav_deploy ~/hn_ws/src/
cd ~/hn_ws && colcon build && source install/setup.bash
# put best_model.zip + vec_normalize.pkl in ~/models, then 3 terminals:
ros2 run hospitalnav_deploy sim_stub
ros2 run hospitalnav_deploy controller --ros-args \
     -p model_zip:=$HOME/models/best_model.zip -p vecnorm_pkl:=$HOME/models/vec_normalize.pkl
ros2 topic echo /cmd_shielded
```
If `/cmd_shielded` streams speed+steering, the brain works. Then follow the guide
for the Pi, SLAM, Nav2, and motors.

## Important: deploy a CURRENT (no-camera) policy
The current benchmark policies are **no-camera**, with a **14-D context** and a
**goal(3)** block. This package targets that observation exactly. (An earlier
version of this scaffold — now folded into this package — was built for a camera
env; the LD19 driver and tests were kept, the camera/12-D obs was retired.)

## The one real blocker
`rrc_bridge.py` needs the **Yahboom RRC-Lite Python SDK** (`set_motor`,
`set_servo_angle`, `get_encoder`) wired in before the wheels turn. Everything else
— LiDAR, SLAM, Nav2, policy, shield, RViz — runs without it.

Deploy `fixed_equal` / `attention` + shield for max success, or **CWN + shield** to
show the live adaptive-weight overlay (`/policy_weights`). See
`docs/real_world_test_protocol.md` for how to run and report the demo honestly.
