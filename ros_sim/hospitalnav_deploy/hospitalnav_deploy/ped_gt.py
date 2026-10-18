"""Ground-truth pedestrian poses for a run: /pedestrian_poses (geometry_msgs/
PoseArray, map frame, ped_routes.LEVELS order) at 10 Hz plus
/pedestrian_markers for RViz. gt_pedestrians turns the array into the policy's
/people channel when people_source:=gt; csv_logger and patrol_sequencer use it
for proximity metrics whatever the source.

SOURCE
  The PedestrianBodies gz system simulates every walker (they yield to the
  robot, so a pose is not a function of sim time alone) and publishes the
  poses it simulated on the gz topic /hospitalnav/pedestrians. This node
  republishes the latest of those. Only if no such message has arrived (a
  world without the plugin) does it fall back to ped_routes' unobstructed
  schedule at sim time, and it says so in the log.
"""
from __future__ import annotations
import math
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray

from .ped_routes import LEVELS, pose_at, PED_RADIUS

GZ_TOPIC = "/hospitalnav/pedestrians"
STALE_S = 1.0        # a gz sample older than this (sim time) is not used


class PedGT(Node):
    def __init__(self):
        super().__init__("ped_gt")
        self.level = str(self.declare_parameter("peds", "none").value)
        self.names = LEVELS.get(self.level, [])
        self.pub = self.create_publisher(PoseArray, "/pedestrian_poses", 10)
        self.mpub = self.create_publisher(MarkerArray, "/pedestrian_markers", 1)
        self.lock = threading.Lock()
        self.latest = None               # (sim t, {name: (x, y, yaw)})
        self.source = None
        self._gz = None
        try:
            from gz.transport13 import Node as GzNode
            from gz.msgs10.pose_v_pb2 import Pose_V
            self._gz = GzNode()
            self._gz.subscribe(Pose_V, GZ_TOPIC, self._gz_cb)
        except Exception as e:                       # pragma: no cover
            self.get_logger().warning(f"gz transport unavailable ({e}); using the schedule")
        self.create_timer(0.1, self._tick)
        self.get_logger().info(f"ped_gt up: level '{self.level}', {len(self.names)} actors")

    def _gz_cb(self, m):
        # gz transport delivery thread: copy and return, never block here
        d = {}
        for p in m.pose:
            q = p.orientation
            d[p.name] = (p.position.x, p.position.y,
                         math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        with self.lock:
            self.latest = (t, d)

    def _poses(self, t):
        with self.lock:
            latest = self.latest
        if latest is not None and abs(t - latest[0]) < STALE_S and all(n in latest[1] for n in self.names):
            src, out = "simulated (PedestrianBodies)", [latest[1][n] for n in self.names]
        else:
            src, out = "schedule (no PedestrianBodies poses)", [pose_at(n, t) for n in self.names]
        if src != self.source:
            self.source = src
            self.get_logger().info(f"pedestrian ground truth: {src}")
        return out

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        arr = PoseArray()
        arr.header.frame_id = "map"
        arr.header.stamp = self.get_clock().now().to_msg()
        marks = MarkerArray()
        for i, (x, y, yaw) in enumerate(self._poses(t)):
            p = Pose()
            p.position.x, p.position.y = float(x), float(y)
            p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            arr.poses.append(p)
            m = Marker()
            m.header = arr.header
            m.ns = "pedestrians"; m.id = i
            m.type = Marker.CYLINDER; m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.85
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2 * PED_RADIUS; m.scale.z = 1.7
            m.color.r, m.color.g, m.color.b, m.color.a = 0.95, 0.55, 0.15, 0.85
            marks.markers.append(m)
        self.pub.publish(arr)
        if marks.markers:
            self.mpub.publish(marks)


def main():
    rclpy.init(); rclpy.spin(PedGT()); rclpy.shutdown()


if __name__ == "__main__":
    main()
