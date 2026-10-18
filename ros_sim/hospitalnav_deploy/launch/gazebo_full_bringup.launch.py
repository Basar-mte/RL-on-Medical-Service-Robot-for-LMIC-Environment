"""Full Gazebo simulation of the LMIC-HospitalNav deployment: SLAM mapping +
Nav2 navigation (global planning off the live SLAM map, no AMCL/map_server) +
the TRAINED RL policy + the safety shield, autonomously patrolling the
corridor/nurse-station/male-ward/female-ward hospital floor (40x30 m).

The policy was trained in training/hospitalnav_lite_env.py (a fast Python
mirror of this exact floor plan — see training/layout.py) and deployed here
unmodified via controller_node.py's normal model_zip/vecnorm_pkl params.

Usage:
  ros2 launch hospitalnav_deploy gazebo_full_bringup.launch.py
  ros2 launch hospitalnav_deploy gazebo_full_bringup.launch.py gui:=true
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory("hospitalnav_deploy")
    gazebo_ros = get_package_share_directory("gazebo_ros")
    slam_toolbox = get_package_share_directory("slam_toolbox")
    nav2_bringup = get_package_share_directory("nav2_bringup")

    default_world = os.path.join(pkg, "worlds", "hospital.world")
    default_zones = os.path.join(pkg, "worlds", "hospital_zones.png")
    urdf_path = os.path.join(pkg, "urdf", "hospitalnav_robot.urdf")
    slam_params = os.path.join(pkg, "config", "slam_toolbox_params.yaml")
    nav2_params = os.path.join(pkg, "config", "nav2_gazebo_sim.yaml")
    # absolute path to this demo's training output — not portable, but this
    # launch file is specific to this machine's checkout, same as the model
    # paths in config/robot.yaml are specific to the real Pi's ~/models.
    training_models = "/home/roky-ubuntu/ros2_ws/hardware_deploy/training/models"
    with open(urdf_path) as f:
        robot_description = f.read()

    args = [
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("x_pose", default_value="37.0"),
        DeclareLaunchArgument("y_pose", default_value="15.0"),
        DeclareLaunchArgument("csv_path", default_value="/tmp/hospitalnav_run.csv"),
        DeclareLaunchArgument("model_zip", default_value=os.path.join(training_models, "best_model.zip")),
        DeclareLaunchArgument("vecnorm_pkl", default_value=os.path.join(training_models, "vec_normalize.pkl")),
        DeclareLaunchArgument("semantic_png", default_value=default_zones),
        DeclareLaunchArgument("semantic_res", default_value="0.05"),
        DeclareLaunchArgument("agent", default_value="fixed_equal"),
        DeclareLaunchArgument("algo", default_value="sac",
                               description="SB3 class of model_zip: sac (default) / td3 / ppo"),
        DeclareLaunchArgument("weight_pt", default_value=""),
        DeclareLaunchArgument("cultural_ratio", default_value="3.0",
                               description="route-planner cultural cost: 3.0 for CARS agents, 0.0 for vanilla/he2025"),
        DeclareLaunchArgument("autonomous", default_value="true",
                               description="true = room_patrol cycles goals forever; false = send one goal manually"),
        DeclareLaunchArgument("patrol_start_delay", default_value="90.0",
                               description="wall-clock seconds after launch before room_patrol sends its first "
                                            "goal, so there's time to arrange/record the Gazebo+RViz windows"),
    ]
    LC = LaunchConfiguration

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, "launch", "gazebo.launch.py")),
        launch_arguments={"world": LC("world"), "gui": LC("gui")}.items())

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_toolbox, "launch", "online_async_launch.py")),
        launch_arguments={"slam_params_file": slam_params, "use_sim_time": "true"}.items())

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup, "launch", "navigation_launch.py")),
        launch_arguments={"params_file": nav2_params, "use_sim_time": "true",
                           "autostart": "true"}.items())

    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description, "use_sim_time": True}]),
        Node(package="gazebo_ros", executable="spawn_entity.py",
             arguments=["-topic", "robot_description", "-entity", "hospitalnav_robot",
                        "-x", LC("x_pose"), "-y", LC("y_pose"), "-z", "0.05",
                        "-timeout", "90"]),
        Node(package="hospitalnav_deploy", executable="people_tracker", name="people_tracker",
             parameters=[{"use_sim_time": True}]),
        # global level of the trained hierarchy: plans on the live SLAM map with the
        # training cost field; takes room_patrol's /patrol_goal (and RViz /goal_pose)
        Node(package="hospitalnav_deploy", executable="route_planner", name="route_planner",
             parameters=[{
                 "cultural_ratio": LC("cultural_ratio"), "semantic_png": LC("semantic_png"),
                 "semantic_res": LC("semantic_res"), "semantic_origin": [0.0, 0.0],
                 "use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="controller", name="hospitalnav_controller",
             parameters=[{
                 "model_zip": LC("model_zip"), "vecnorm_pkl": LC("vecnorm_pkl"),
                 "agent": LC("agent"), "algo": LC("algo"), "weight_pt": LC("weight_pt"),
                 "semantic_png": LC("semantic_png"), "semantic_res": LC("semantic_res"),
                 "semantic_origin": [0.0, 0.0], "map_w": 40.0, "map_h": 30.0,
                 "use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="csv_logger", name="csv_logger",
             parameters=[{"csv_path": LC("csv_path"), "use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="sim_markers", name="sim_markers",
             parameters=[{"x_pose": LC("x_pose"), "y_pose": LC("y_pose"), "use_sim_time": True}]),
    ]

    # room_patrol starts on a real (wall-clock) delay — gzserver/gzclient/SLAM/
    # Nav2 all come up immediately, but the robot doesn't move until this
    # fires, giving time to open/arrange RViz and start recording first.
    room_patrol_delayed = TimerAction(
        period=LC("patrol_start_delay"),
        actions=[Node(package="hospitalnav_deploy", executable="room_patrol", name="room_patrol",
                       parameters=[{"use_sim_time": True}])])

    return LaunchDescription(args + [gazebo, slam, navigation] + nodes + [room_patrol_delayed])
