"""Gazebo simulation of the LMIC-HospitalNav deployment: spawns the robot in a
hospital-like world, runs SLAM (mapping) live, and drives it with the safety
shield active. Stage 1 of the sim path: no Nav2 yet, /rl_goal comes from a
fixed goal_publisher instead of Nav2's global plan (see
gazebo_full_bringup.launch.py for the Nav2 version).

Usage:
  ros2 launch hospitalnav_deploy gazebo_slam_bringup.launch.py
  ros2 launch hospitalnav_deploy gazebo_slam_bringup.launch.py gui:=true
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory("hospitalnav_deploy")
    gazebo_ros = get_package_share_directory("gazebo_ros")
    slam_toolbox = get_package_share_directory("slam_toolbox")

    default_world = os.path.join(pkg, "worlds", "hospital.world")
    urdf_path = os.path.join(pkg, "urdf", "hospitalnav_robot.urdf")
    slam_params = os.path.join(pkg, "config", "slam_toolbox_params.yaml")
    with open(urdf_path) as f:
        robot_description = f.read()

    args = [
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("x_pose", default_value="0.0"),
        DeclareLaunchArgument("y_pose", default_value="-1.5"),
        DeclareLaunchArgument("goal_x", default_value="3.5"),
        DeclareLaunchArgument("goal_y", default_value="-1.0"),
        DeclareLaunchArgument("csv_path", default_value="/tmp/hospitalnav_run.csv"),
    ]
    LC = LaunchConfiguration

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, "launch", "gazebo.launch.py")),
        launch_arguments={"world": LC("world"), "gui": LC("gui")}.items())

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_toolbox, "launch", "online_async_launch.py")),
        launch_arguments={"slam_params_file": slam_params, "use_sim_time": "true"}.items())

    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description, "use_sim_time": True}]),
        Node(package="gazebo_ros", executable="spawn_entity.py",
             arguments=["-topic", "robot_description", "-entity", "hospitalnav_robot",
                        "-x", LC("x_pose"), "-y", LC("y_pose"), "-z", "0.05",
                        "-timeout", "90"]),
        Node(package="hospitalnav_deploy", executable="people_tracker", name="people_tracker",
             parameters=[{"use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="goal_publisher", name="goal_publisher",
             parameters=[{"goal_x": LC("goal_x"), "goal_y": LC("goal_y"), "use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="placeholder_controller",
             name="hospitalnav_controller", parameters=[{"use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="csv_logger", name="csv_logger",
             parameters=[{"csv_path": LC("csv_path"), "use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="sim_markers", name="sim_markers",
             parameters=[{"x_pose": LC("x_pose"), "y_pose": LC("y_pose"),
                          "static_goal": True, "goal_x": LC("goal_x"), "goal_y": LC("goal_y"),
                          "use_sim_time": True}]),
    ]
    return LaunchDescription(args + [gazebo, slam] + nodes)
