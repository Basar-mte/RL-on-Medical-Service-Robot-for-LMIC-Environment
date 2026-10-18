"""Bring up Gazebo (gz-sim) with the hospital world + the Ackermann robot, bridge
the topics to ROS 2, and run the trained policy + shield - NO retraining.

Targets ros_gz (Gazebo Fortress/Harmonic, the pairing for ROS 2 Humble). Install:
  sudo apt install ros-humble-ros-gz ros-humble-ros-gz-bridge ros-humble-ros-gz-sim

Run (after 'colcon build' of hospitalnav_deploy and generating the world):
  ros2 launch <this file> \
     model_zip:=$HOME/models/best_model.zip vecnorm_pkl:=$HOME/models/vec_normalize.pkl \
     agent:=cwn weight_pt:=$HOME/models/cwn_best.pt \
     semantic_png:=$HOME/hn_ws/maps/hospital_zones.png

The robot drives toward /rl_goal using the policy; the shield keeps it safe.
For a crowd, add <actor>s to the world and bridge their poses to /pedestrian_poses,
then run 'gt_pedestrians' (see README). Without it, /people is zeros (no crowd).
"""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

HERE = Path(__file__).resolve().parent.parent      # hardware_deploy/gazebo
WORLD = str(HERE / "worlds" / "hospital.world")
URDF = str(HERE / "urdf" / "hospitalbot.urdf")


def generate_launch_description():
    LC = LaunchConfiguration
    args = [DeclareLaunchArgument(a, default_value=d) for a, d in [
        ("model_zip", ""), ("vecnorm_pkl", ""), ("weight_pt", ""), ("agent", "cwn"),
        ("semantic_png", ""), ("goal_x", "8.0"), ("goal_y", "4.0")]]  # e.g. emergency dept

    ros_gz_sim = get_package_share_directory("ros_gz_sim")
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-r {WORLD}"}.items())

    spawn = Node(package="ros_gz_sim", executable="create",
                 arguments=["-file", URDF, "-name", "hospitalbot", "-x", "20", "-y", "13.5", "-z", "0.1"],
                 output="screen")

    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  arguments=[
                      "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                      "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                      "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                      "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                  ])

    map_odom = Node(package="tf2_ros", executable="static_transform_publisher",
                    arguments=["0", "0", "0", "0", "0", "0", "map", "odom"])

    adapter = Node(package="hospitalnav_deploy", executable="gz_cmd_adapter")
    controller = Node(package="hospitalnav_deploy", executable="controller",
                      parameters=[{"model_zip": LC("model_zip"), "vecnorm_pkl": LC("vecnorm_pkl"),
                                   "weight_pt": LC("weight_pt"), "agent": LC("agent"),
                                   "semantic_png": LC("semantic_png"), "map_w": 40.0, "map_h": 30.0}])

    goal = ExecuteProcess(cmd=[
        "ros2", "topic", "pub", "-r", "1", "/rl_goal", "geometry_msgs/msg/PointStamped",
        ["{header: {frame_id: map}, point: {x: ", LC("goal_x"), ", y: ", LC("goal_y"), ", z: 0.0}}"]],
        output="screen")

    return LaunchDescription(args + [gz, spawn, bridge, map_odom, adapter, controller, goal])
