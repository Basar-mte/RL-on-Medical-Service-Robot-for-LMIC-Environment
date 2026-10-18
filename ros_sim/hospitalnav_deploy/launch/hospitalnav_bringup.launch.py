"""Runtime bring-up for the hierarchical RL stack (after the map is built):
Nav2 localization + the route planner (global level, same cost field as training)
+ the trained RL policy (local level) + the safety shield.

Usage:
  ros2 launch hospitalnav_deploy hospitalnav_bringup.launch.py \
      map:=$HOME/hn_ws/maps/hospital.yaml \
      model_zip:=$HOME/models/final_model.zip vecnorm_pkl:=$HOME/models/vec_normalize.pkl \
      weight_pt:=$HOME/models/attention_final.pt agent:=attention \
      semantic_png:=$HOME/hn_ws/maps/hospital_zones.png semantic_res:=0.05 \
      cultural_ratio:=3.0        # 0.0 for the culture-blind vanilla / he2025 policies
Then in RViz: set the initial pose (2D Pose Estimate) and click "2D Goal Pose"
(/goal_pose) — or run goal_publisher with -p topic:=/nav_goal. The planner publishes
the waypoint + remaining route length on /rl_goal and the route on /rl_route.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    pkg = FindPackageShare("hospitalnav_deploy")
    nav2 = FindPackageShare("nav2_bringup")
    params = PathJoinSubstitution([pkg, "config", "nav2_hospitalnav.yaml"])

    args = [DeclareLaunchArgument(a, default_value=d) for a, d in [
        ("map", ""), ("model_zip", ""), ("vecnorm_pkl", ""), ("weight_pt", ""),
        ("agent", "cwn"), ("semantic_png", ""), ("semantic_res", "0.05"), ("urgency", "0.9"),
        ("tod_hour", "12.0"), ("bearing_offset", "0.0"), ("map_w", "40.0"), ("map_h", "30.0"),
        ("cultural_ratio", "3.0"), ("lookahead", "2.5")]]
    LC = LaunchConfiguration

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2, "/launch/localization_launch.py"]),
        launch_arguments={"map": LC("map"), "params_file": params}.items())
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2, "/launch/navigation_launch.py"]),
        launch_arguments={"params_file": params, "use_sim_time": "false"}.items())

    nodes = [
        Node(package="hospitalnav_deploy", executable="rrc_bridge", name="rrc_bridge"),
        Node(package="hospitalnav_deploy", executable="people_tracker", name="people_tracker"),
        # global level: route planner on the localization map (replaces path_to_waypoint)
        Node(package="hospitalnav_deploy", executable="route_planner", name="route_planner",
             parameters=[{
                 "lookahead": LC("lookahead"), "cultural_ratio": LC("cultural_ratio"),
                 "tod_hour": LC("tod_hour"), "semantic_png": LC("semantic_png"),
                 "semantic_res": LC("semantic_res")}]),
        Node(package="hospitalnav_deploy", executable="controller", name="hospitalnav_controller",
             parameters=[{
                 "model_zip": LC("model_zip"), "vecnorm_pkl": LC("vecnorm_pkl"),
                 "weight_pt": LC("weight_pt"), "agent": LC("agent"),
                 "semantic_png": LC("semantic_png"), "semantic_res": LC("semantic_res"),
                 "urgency": LC("urgency"), "tod_hour": LC("tod_hour"),
                 "bearing_offset": LC("bearing_offset"),
                 "map_w": LC("map_w"), "map_h": LC("map_h")}]),
        # static LiDAR mount transform (edit x,y,z,yaw to your robot)
        Node(package="tf2_ros", executable="static_transform_publisher",
             arguments=["0.15", "0", "0.2", "0", "0", "0", "base_link", "laser"]),
    ]
    return LaunchDescription(args + [localization, navigation] + nodes)
