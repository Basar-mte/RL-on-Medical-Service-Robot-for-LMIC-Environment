import os
from setuptools import setup
from glob import glob

package_name = "hospitalnav_deploy"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/urdf", glob("urdf/*.urdf")),
        (f"share/{package_name}/worlds", glob("worlds/*.world") + glob("worlds/*.png")
         + glob("worlds/*.yaml") + glob("worlds/*.pgm")),
        # every mesh directory (acker = robot, fuel = vendored assets, gen = textured boxes)
        *[(f"share/{package_name}/{d}", [os.path.join(d, f) for f in fs])
          for d, _sub, fs in os.walk("meshes") if fs],
        (f"share/{package_name}/materials/textures", glob("materials/textures/*.png")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    tests_require=["pytest"],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Md. Abul Basar Roky",
    maintainer_email="m.a.basar.mte@gmail.com",
    description="LMIC-HospitalNav-v1 hardware deployment: RL policy + Nav2 + safety shield.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "shield_node = hospitalnav_deploy.shield_node:main",
            "people_tracker = hospitalnav_deploy.perception:main",
            "path_to_waypoint = hospitalnav_deploy.path_to_waypoint:main",
            "route_planner = hospitalnav_deploy.route_planner_node:main",
            "controller = hospitalnav_deploy.controller_node:main",
            "rrc_bridge = hospitalnav_deploy.rrc_bridge:main",
            "sim_stub = hospitalnav_deploy.sim_stub:main",
            "lidar_node = hospitalnav_deploy.lidar_node:main",
            "placeholder_controller = hospitalnav_deploy.placeholder_controller:main",
            "csv_logger = hospitalnav_deploy.csv_logger:main",
            "goal_publisher = hospitalnav_deploy.goal_publisher:main",
            "nav2_goal_sender = hospitalnav_deploy.nav2_goal_sender:main",
            "room_patrol = hospitalnav_deploy.room_patrol:main",
            "patrol_sequencer = hospitalnav_deploy.patrol_sequencer:main",
            "gz_cmd_adapter = hospitalnav_deploy.gz_cmd_adapter:main",
            "odom_tf_broadcaster = hospitalnav_deploy.odom_tf_broadcaster:main",
            "gt_pedestrians = hospitalnav_deploy.gt_pedestrians:main",
            "sim_markers = hospitalnav_deploy.sim_markers:main",
            "ped_gt = hospitalnav_deploy.ped_gt:main",
        ],
    },
)
