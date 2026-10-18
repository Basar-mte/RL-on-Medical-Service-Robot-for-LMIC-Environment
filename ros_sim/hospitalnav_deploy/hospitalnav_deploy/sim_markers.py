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
from nav_msgs.msg import OccupancyGrid, Path
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
        # the walls as markers (see _on_map); off for the live-mapping view,
        # where the point is to watch the SLAM map appear
        self.draw_map_cells = bool(gp("draw_map_cells", True).value)
        self.label_size = float(gp("label_size", 0.8).value)
        self.goal_x = float(gp("goal_x", 0.0).value)
        self.goal_y = float(gp("goal_y", 0.0).value)

        self.goal = (self.goal_x, self.goal_y) if self.static_goal else None
        self.trail = []                 # [(x, y), ...] traveled path, map frame
        self.pose = None                # (x, y, yaw) for the footprint marker
        # The occupancy map as a CUBE_LIST marker: rviz2's own Map display
        # draws nothing on this machine (its indexed_8bit_image GLSL shader
        # fails on the Intel/Mesa context, measured 2026-10-16), while plain
        # markers render fine - so the walls the robot navigates by are drawn
        # here, one 5 cm cube per occupied cell.
        self.map_cells = None
        # the planner's route as first planned for the current goal: /rl_route
        # shrinks to nothing as the robot arrives, so a figure taken at the end
        # of a run would otherwise show no route at all
        self.planned = []
        self.planned_goal = None
        self.create_subscription(Path, "/rl_route", self._on_route, 10)
        self.tf = Buffer(); TransformListener(self.tf, self)
        self.pub = self.create_publisher(MarkerArray, "/sim_markers", LATCHED)
        self.create_subscription(PoseStamped, "/patrol_goal", self._on_goal, 10)
        if self.draw_map_cells:
            self.create_subscription(OccupancyGrid, "/map", self._on_map, LATCHED)
        self.create_timer(0.1, self._track)
        self.create_timer(0.2, self._publish)
        self.get_logger().info("sim_markers up: room labels + start/goal markers + trail")

    def _on_goal(self, msg: PoseStamped):
        self.goal = (msg.pose.position.x, msg.pose.position.y)

    def _on_route(self, msg: Path):
        if len(msg.poses) < 2:
            return
        end = (msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y)
        if self.planned_goal is None or math.hypot(end[0] - self.planned_goal[0], end[1] - self.planned_goal[1]) > 0.5:
            self.planned_goal = end
            self.planned = [(q.pose.position.x, q.pose.position.y) for q in msg.poses]

    def _planned_marker(self):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "planned_route"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 0.08, 0.39, 0.86, 0.85     # planned route: blue
        m.points = [Point(x=x, y=y, z=0.02) for x, y in self.planned]
        return m

    def _on_map(self, m: OccupancyGrid):
        res, ox, oy, w = m.info.resolution, m.info.origin.position.x, m.info.origin.position.y, m.info.width
        self.map_cells = [Point(x=ox + (i % w + 0.5) * res, y=oy + (i // w + 0.5) * res, z=0.0)
                          for i, v in enumerate(m.data) if v >= 65]
        self.map_res = res

    def _map_marker(self):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "map_cells"; m.id = 0
        m.type = Marker.CUBE_LIST; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = self.map_res
        m.color.r, m.color.g, m.color.b, m.color.a = 0.15, 0.15, 0.17, 1.0
        m.points = self.map_cells
        return m

    def _track(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return
        x, y = t.transform.translation.x, t.transform.translation.y
        q = t.transform.rotation
        self.pose = (x, y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.05:
            self.trail.append((x, y))

    def _text(self, i, label, x, y, r, g, b):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "room_labels"; m.id = i
        m.type = Marker.TEXT_VIEW_FACING; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 1.6
        m.pose.orientation.w = 1.0
        m.scale.z = self.label_size
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
        m.scale.x = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 0.86, 0.08, 0.10, 1.0   # traveled path: red
        m.points = [Point(x=x, y=y, z=0.03) for x, y in self.trail]
        return m

    def _footprint(self):
        """The robot as a figure reads it: a dark footprint (MentorPi body,
        0.30 x 0.20 m) and a heading arrow, drawn above every other layer."""
        x, y, yaw = self.pose
        body = Marker()
        body.header.frame_id = "map"
        body.ns = "robot_footprint"; body.id = 0
        body.type = Marker.CUBE; body.action = Marker.ADD
        body.pose.position.x, body.pose.position.y, body.pose.position.z = x, y, 0.35
        body.pose.orientation.z, body.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        body.scale.x, body.scale.y, body.scale.z = 0.30, 0.20, 0.05
        body.color.r, body.color.g, body.color.b, body.color.a = 0.10, 0.10, 0.12, 1.0
        arrow = Marker()
        arrow.header.frame_id = "map"
        arrow.ns = "robot_heading"; arrow.id = 0
        arrow.type = Marker.ARROW; arrow.action = Marker.ADD
        arrow.points = [Point(x=x, y=y, z=0.40),
                        Point(x=x + 0.45 * math.cos(yaw), y=y + 0.45 * math.sin(yaw), z=0.40)]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.06, 0.14, 0.14
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.80, 0.05, 1.0
        return [body, arrow]

    def _publish(self):
        arr = MarkerArray()
        if self.map_cells:
            arr.markers.append(self._map_marker())
        for i, (label, x, y, r, g, b) in enumerate(ROOMS):
            arr.markers.append(self._text(i, label, x, y, r, g, b))
        arr.markers.append(self._sphere("start_point", 0, self.x_pose, self.y_pose,
                                         0.1, 0.85, 0.15))
        if self.goal is not None:
            gx, gy = self.goal
            arr.markers.append(self._sphere("goal_point", 0, gx, gy, 0.10, 0.35, 0.85))
        if self.planned:
            arr.markers.append(self._planned_marker())
        if self.trail:
            arr.markers.append(self._trail_marker())
        if self.pose is not None:
            arr.markers.extend(self._footprint())
        self.pub.publish(arr)


def main():
    rclpy.init(); rclpy.spin(SimMarkers()); rclpy.shutdown()


if __name__ == "__main__":
    main()
