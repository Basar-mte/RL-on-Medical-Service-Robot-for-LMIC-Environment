"""RViz-only visualization helper: text labels for the 6 hospital rooms + the
nurse desk, a marker at the robot's spawn point, a live marker tracking the
current Nav2/patrol goal, and a growing breadcrumb trail of the robot's
actual traveled path (map frame, via TF — distinct from Nav2's /plan, which
only shows the current planned route, not where the robot has actually
been). Purely cosmetic — publishes visualization_msgs/MarkerArray, never
touches /scan, /odom, or any topic the RL policy or Nav2 consume.

Room label positions/colors mirror training/layout.py's ROOMS + ROOM_COLOR
(duplicated here rather than imported: this package and training/ are
separate, unpackaged Python trees with no shared import path).
"""
from __future__ import annotations
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener

# (label, x, y, r, g, b) — centres/colors match training/layout.py's ROOMS/ROOM_COLOR.
ROOMS = [
    ("Male Ward",        6.5,  23.5, 0.25, 0.45, 0.90),
    ("General Ward 1",   20.0, 23.5, 0.30, 0.80, 0.45),
    ("Female Ward",      33.5, 23.5, 0.90, 0.35, 0.60),
    ("Emergency Room",   5.0,  7.5,  0.95, 0.25, 0.15),
    ("Prayer Room",      13.0, 7.5,  0.85, 0.75, 0.20),
    ("General Ward 3",   23.5, 7.5,  0.45, 0.65, 0.85),
    ("General Ward 2",   31.0, 7.5,  0.55, 0.35, 0.85),
    ("Nurse Desk",       37.0, 16.4,  0.55, 0.35, 0.20),
]

LATCHED = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


class SimMarkers(Node):
    def __init__(self):
        super().__init__("sim_markers")
        gp = self.declare_parameter
        self.x_pose = float(gp("x_pose", 37.0).value)
        self.y_pose = float(gp("y_pose", 15.0).value)
        self.static_goal = bool(gp("static_goal", False).value)
        self.goal_x = float(gp("goal_x", 0.0).value)
        self.goal_y = float(gp("goal_y", 0.0).value)

        self.goal = (self.goal_x, self.goal_y) if self.static_goal else None
        self.trail = []                 # [(x, y), ...] traveled path, map frame
        self.tf = Buffer(); TransformListener(self.tf, self)
        self.pub = self.create_publisher(MarkerArray, "/sim_markers", LATCHED)
        self.create_subscription(PoseStamped, "/patrol_goal", self._on_goal, 10)
        self.create_timer(0.5, self._track)
        self.create_timer(1.0, self._publish)
        self.get_logger().info("sim_markers up: room labels + start/goal markers + trail")

    def _on_goal(self, msg: PoseStamped):
        self.goal = (msg.pose.position.x, msg.pose.position.y)

    def _track(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return
        x, y = t.transform.translation.x, t.transform.translation.y
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.05:
            self.trail.append((x, y))

    def _text(self, i, label, x, y, r, g, b):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "room_labels"; m.id = i
        m.type = Marker.TEXT_VIEW_FACING; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 1.6
        m.pose.orientation.w = 1.0
        m.scale.z = 0.8
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
        m.text = label
        return m

    def _sphere(self, ns, i, x, y, r, g, b):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = ns; m.id = i
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.3
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.35
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
        return m

    def _trail_marker(self):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "traveled_path"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.65, 0.0, 1.0
        m.points = [Point(x=x, y=y, z=0.05) for x, y in self.trail]
        return m

    def _publish(self):
        arr = MarkerArray()
        for i, (label, x, y, r, g, b) in enumerate(ROOMS):
            arr.markers.append(self._text(i, label, x, y, r, g, b))
        arr.markers.append(self._sphere("start_point", 0, self.x_pose, self.y_pose,
                                         0.1, 0.85, 0.15))
        if self.goal is not None:
            gx, gy = self.goal
            arr.markers.append(self._sphere("goal_point", 0, gx, gy, 0.9, 0.1, 0.1))
        if self.trail:
            arr.markers.append(self._trail_marker())
        self.pub.publish(arr)


def main():
    rclpy.init(); rclpy.spin(SimMarkers()); rclpy.shutdown()


if __name__ == "__main__":
    main()
