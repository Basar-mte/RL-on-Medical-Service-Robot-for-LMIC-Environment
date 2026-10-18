"""Bridges Nav2's global path to the RL policy's goal (alternative to
route_planner_node when Nav2's planner is preferred; it cannot apply the
cultural route costs, so use route_planner for the CARS agents).

Subscribes: /plan (nav_msgs/Path) from Nav2's planner_server, and the robot pose
(via TF map->base_link). Publishes /rl_goal (geometry_msgs/PointStamped, map
frame): a lookahead point LOOKAHEAD m ahead on the path, with z = the remaining
path length from the robot (the hierarchical policies' distance channel). The
controller_node turns this into the 3-D goal observation.
"""
from __future__ import annotations
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener

LOOKAHEAD = 2.5   # m ahead on the global path — matches the trained planner (experiments/planner)


class PathToWaypoint(Node):
    def __init__(self):
        super().__init__("path_to_waypoint")
        self.path = []
        self.tf = Buffer(); TransformListener(self.tf, self)
        self.create_subscription(Path, "/plan", self._plan, 10)
        self.pub = self.create_publisher(PointStamped, "/rl_goal", 10)
        self.create_timer(0.1, self._tick)

    def _plan(self, msg: Path):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _robot_xy(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def _tick(self):
        if not self.path:
            return
        rc = self._robot_xy()
        if rc is None:
            return
        rx, ry = rc
        # nearest point on the path, then walk forward LOOKAHEAD metres
        i0 = min(range(len(self.path)), key=lambda i: (self.path[i][0]-rx)**2 + (self.path[i][1]-ry)**2)
        acc, wp = 0.0, None
        total = math.hypot(self.path[i0][0]-rx, self.path[i0][1]-ry)   # remaining path length from the robot
        for i in range(i0, len(self.path)-1):
            (ax, ay), (bx, by) = self.path[i], self.path[i+1]
            seg = math.hypot(bx-ax, by-ay); acc += seg; total += seg
            if wp is None and acc >= LOOKAHEAD:
                wp = (bx, by)                      # first point LOOKAHEAD m ahead; keep summing total
        gx, gy = wp if wp is not None else self.path[-1]
        m = PointStamped(); m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.point.x, m.point.y, m.point.z = gx, gy, max(total, 1e-3)
        self.pub.publish(m)


def main():
    rclpy.init(); rclpy.spin(PathToWaypoint()); rclpy.shutdown()


if __name__ == "__main__":
    main()
