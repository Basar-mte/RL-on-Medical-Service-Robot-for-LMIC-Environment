"""gz-sim (Fortress/Harmonic) minimal demo: SLAM mapping + Nav2 navigation +
the trained RL policy + the safety shield, driving to ONE fixed goal in a
small corridor + single-ward hospital generated from
configs/dmch_hospital.yaml. Same trained checkpoint as the classic-Gazebo
version — NO retraining; only the simulator differs.

Prereqs: sudo apt install ros-humble-ros-gz ros-humble-ros-gz-bridge ros-humble-ros-gz-sim
Run: python3 gen_gazebo_world.py && python3 gen_zone_png.py  (once, or after
editing configs/dmch_hospital.yaml), then:
  ros2 launch <this file> gui:=true
"""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

HERE = Path(__file__).resolve().parent.parent          # hardware_deploy/gazebo
WORLD = str(HERE / "worlds" / "hospital.world")
URDF = str(HERE / "urdf" / "hospitalbot.urdf")
DEFAULT_ZONES = str(HERE / "worlds" / "hospital_zones.png")
TRAINING_MODELS = "/home/roky/RL project/hardware_deploy/training/models"


def generate_launch_description():
    LC = LaunchConfiguration
    pkg = get_package_share_directory("hospitalnav_deploy")
    slam_toolbox = get_package_share_directory("slam_toolbox")
    nav2_bringup = get_package_share_directory("nav2_bringup")
    ros_gz_sim = get_package_share_directory("ros_gz_sim")

    slam_params = os.path.join(pkg, "config", "slam_toolbox_params.yaml")
    nav2_params = os.path.join(pkg, "config", "nav2_gazebo_sim.yaml")

    args = [
        # headless by default — the software-rendered GUI viewport alone costs
        # ~170% CPU on this machine's GL stack; pass gui:=true to watch it.
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("x_pose", default_value="20.0"),
        DeclareLaunchArgument("y_pose", default_value="14.0"),
        DeclareLaunchArgument("csv_path", default_value="/tmp/hospitalnav_run.csv"),
        DeclareLaunchArgument("model_zip", default_value=os.path.join(TRAINING_MODELS, "best_model.zip")),
        DeclareLaunchArgument("vecnorm_pkl", default_value=os.path.join(TRAINING_MODELS, "vec_normalize.pkl")),
        DeclareLaunchArgument("semantic_png", default_value=DEFAULT_ZONES),
        # just past the ward doorway (spawn is (20,14), doorway gap is x=[18,20]
        # at y=17) — close enough that Nav2 can plan to it before the SLAM map
        # has grown much, instead of "Goal pose is out of costmap!" forever.
        DeclareLaunchArgument("goal_x", default_value="19.0"),
        DeclareLaunchArgument("goal_y", default_value="19.5"),
    ]

    gz_server_flag = PythonExpression(["'' if '", LC("gui"), "' == 'true' else '-s'"])
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")),
        # gz_sim.launch.py runs this through a shell (shell=True) — the world
        # path must be quoted since this checkout's directory has a space in it.
        # --render-engine-gui ogre: the GUI's default ogre2 client hangs (Qt/
        # Wayland-specific, separate from the sensors-side Ogre2/GL3Plus crash
        # fixed by LIBGL_ALWAYS_SOFTWARE below) — ogre1 renders the viewport fine.
        launch_arguments={"gz_args": [
            "-r ", gz_server_flag, " --render-engine-gui ogre",
            ' "', WORLD, '"']}.items())

    spawn = Node(package="ros_gz_sim", executable="create",
                 arguments=["-file", URDF, "-name", "hospitalbot",
                            "-x", LC("x_pose"), "-y", LC("y_pose"), "-z", "0.1"],
                 output="screen")

    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  arguments=[
                      "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                      "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                      "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                      "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                  ])

    laser_tf = Node(package="tf2_ros", executable="static_transform_publisher",
                    arguments=["0.15", "0", "0.12", "0", "0", "0", "base_link", "laser"])

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(slam_toolbox, "launch", "online_async_launch.py")),
        launch_arguments={"slam_params_file": slam_params, "use_sim_time": "true"}.items())

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup, "launch", "navigation_launch.py")),
        launch_arguments={"params_file": nav2_params, "use_sim_time": "true",
                           "autostart": "true"}.items())

    nodes = [
        Node(package="hospitalnav_deploy", executable="gz_cmd_adapter", name="gz_cmd_adapter"),
        Node(package="hospitalnav_deploy", executable="odom_tf_broadcaster", name="odom_tf_broadcaster",
             parameters=[{"spawn_x": LC("x_pose"), "spawn_y": LC("y_pose")}]),
        Node(package="hospitalnav_deploy", executable="path_to_waypoint", name="path_to_waypoint",
             parameters=[{"use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="controller", name="hospitalnav_controller",
             parameters=[{
                 "model_zip": LC("model_zip"), "vecnorm_pkl": LC("vecnorm_pkl"),
                 "agent": "fixed_equal", "semantic_png": LC("semantic_png"),
                 "semantic_res": 0.05, "semantic_origin": [0.0, 0.0],
                 "map_w": 40.0, "map_h": 30.0, "use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="csv_logger", name="csv_logger",
             parameters=[{"csv_path": LC("csv_path"), "use_sim_time": True}]),
        Node(package="hospitalnav_deploy", executable="nav2_goal_sender", name="nav2_goal_sender",
             parameters=[{"goal_x": LC("goal_x"), "goal_y": LC("goal_y"), "use_sim_time": True}]),
    ]
    # ogre2/GL3Plus throws Ogre::UnimplementedException (mipmap/texture-copy op
    # unimplemented) on this machine's virtualized GL stack — forcing full
    # software rendering (llvmpipe) avoids the broken hybrid GL path and, as
    # a bonus, is what actually made gpu_lidar return real ranges instead of
    # silently-wrong all-minimum readings.
    force_software_gl = SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1")

    return LaunchDescription(
        [force_software_gl] + args + [gz, spawn, bridge, laser_tf, slam, navigation] + nodes)
