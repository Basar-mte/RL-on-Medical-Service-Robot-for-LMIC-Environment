"""Writes the publication RViz configs into hospitalnav_deploy/rviz/:

  paper_topdown.rviz   4K top-down floor plan: prebuilt map, cultural route
                       (blue), travelled path (RED), LD19 scan (teal),
                       pedestrians (orange), robot, room labels
  paper_orbit.rviz     4K 3D chase view of the robot with the same layers
  paper_slam.rviz      4K top-down of the LIVE SLAM map (/slam_map, viz_slam:=true)
  video_mapping.rviz   the live-SLAM inset of the demo video: /slam_map as it
                       is built, same overlays, 1200 x 900

No Panels section, so the 3D view fills the window; the background is the
sentinel grey 254 so capture_rviz.py can crop exactly to the render area.

    python3 make_rviz_configs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "hospitalnav_deploy", "rviz")

DISPLAYS = """
    - Class: rviz_default_plugins/Map
      Name: Map
      Enabled: true
      Topic:
        Value: {map_topic}
        Durability Policy: Transient Local
        Reliability Policy: Reliable
      Color Scheme: map
      Alpha: 1.0
      Draw Behind: true
    - Class: rviz_default_plugins/MarkerArray
      Name: Room labels, start/goal, travelled path
      Enabled: true
      Topic:
        Value: /sim_markers
        Durability Policy: Transient Local
    - Class: rviz_default_plugins/MarkerArray
      Name: Pedestrians
      Enabled: true
      Topic:
        Value: /pedestrian_markers
        Durability Policy: Volatile
    - Class: rviz_default_plugins/Path
      Name: Cultural route
      Enabled: true
      Topic:
        Value: /rl_route
        Durability Policy: Volatile
      Color: 20; 100; 220
      Line Style: Billboards
      Line Width: {route_w}
      Alpha: 0.9
    - Class: rviz_default_plugins/PointStamped
      Name: Policy waypoint
      Enabled: true
      Topic:
        Value: /rl_goal
      Color: 130; 60; 200
      Radius: {wp_r}
      Alpha: 1.0
    - Class: rviz_default_plugins/LaserScan
      Name: LD19 scan
      Enabled: true
      Topic:
        Value: /scan
        Reliability Policy: Best Effort
      Color: 0; 150; 190
      Color Transformer: FlatColor
      Size (m): {scan_size}
      Size (Pixels): 3
      Style: {scan_style}
      Alpha: 1.0
      Decay Time: 0
    - Class: rviz_default_plugins/RobotModel
      Name: MentorPi
      Enabled: true
      Description Source: Topic
      Description Topic:
        Value: /robot_description
        Durability Policy: Transient Local
      Alpha: 1.0
      Visual Enabled: true
      Collision Enabled: false
"""

TEMPLATE = """Panels: []
Visualization Manager:
  Class: ""
  Global Options:
    Background Color: 254; 254; 254
    Fixed Frame: map
    Frame Rate: 30
  Displays:{displays}
  Tools:
    - Class: rviz_default_plugins/MoveCamera
  Views:
    Current:
{view}
Window Geometry:
  Height: {h}
  Width: {w}
  X: 0
  Y: 0
  Hide Left Dock: true
  Hide Right Dock: true
"""

TOPDOWN = """      Class: rviz_default_plugins/TopDownOrtho
      Name: Floor plan
      Target Frame: map
      X: 20
      Y: 15
      Scale: {scale}
      Angle: 0"""

ORBIT = """      Class: rviz_default_plugins/Orbit
      Name: Chase
      Target Frame: base_link
      Distance: 6.5
      Focal Point:
        X: 0.8
        Y: 0.0
        Z: 0.0
      Pitch: 0.95
      Yaw: 3.1416
      Near Clip Distance: 0.01"""

LOCAL = """      Class: rviz_default_plugins/TopDownOrtho
      Name: Around the robot
      Target Frame: base_link
      X: 0.8
      Y: 0
      Scale: 300
      Angle: 0"""

CONFIGS = {
    "paper_local.rviz": dict(map_topic="/map", route_w=0.06, wp_r=0.10, scan_size=0.07,
                             scan_style="Flat Squares", view=LOCAL, w=3840, h=2160),
    "paper_topdown.rviz": dict(map_topic="/map", route_w=0.14, wp_r=0.25, scan_size=0.11,
                               scan_style="Flat Squares", view=TOPDOWN.format(scale=64), w=3840, h=2160),
    "paper_orbit.rviz": dict(map_topic="/map", route_w=0.10, wp_r=0.18, scan_size=0.05,
                             scan_style="Spheres", view=ORBIT, w=3840, h=2160),
    "paper_slam.rviz": dict(map_topic="/slam_map", route_w=0.14, wp_r=0.25, scan_size=0.11,
                            scan_style="Flat Squares", view=TOPDOWN.format(scale=64), w=3840, h=2160),
    "video_mapping.rviz": dict(map_topic="/slam_map", route_w=0.18, wp_r=0.30, scan_size=0.18,
                               scan_style="Flat Squares", view=TOPDOWN.format(scale=27), w=1200, h=930),
}

if __name__ == "__main__":
    for name, c in CONFIGS.items():
        d = DISPLAYS.format(**{k: c[k] for k in ("map_topic", "route_w", "wp_r", "scan_size", "scan_style")})
        text = TEMPLATE.format(displays=d, view=c["view"], w=c["w"], h=c["h"])
        with open(os.path.join(OUT, name), "w") as f:
            f.write(text)
        print("wrote", os.path.join("hospitalnav_deploy/rviz", name))
