"""PLACEHOLDER controller for the Gazebo pipeline demo — NOT the trained SAC/CWN
policy. No trained checkpoint (best_model.zip / vec_normalize.pkl) was available
on this machine, so this node stands in for controller_node.py: it steers toward
/rl_goal with a simple proportional heuristic, then applies the SAME safety shield
(scan_utils.shield) as the real deployment. Swap back to controller_node once you
have a real checkpoint — its topic/param interface is unchanged.

Subscribes: /scan, /odom, /rl_goal (PointStamped, map), TF map->base_link
Publishes:  /cmd_shielded (Twist), /policy_weights (Float32MultiArray, dummy [1,0,0,0])
"""
from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PointStamped
from tf2_ros import Buffer, TransformListener
from .scan_utils import scan_to_lidar48, shield, N_RAYS, DT, MAX_LINEAR


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class PlaceholderController(Node):
    def __init__(self):
        super().__init__("hospitalnav_placeholder_controller")
        self.offset = float(self.declare_parameter("bearing_offset", 0.0).value)
        self.lidar = np.ones(N_RAYS, np.float32)
        self.goal = None
        self.tf = Buffer()
        TransformListener(self.tf, self)

        self.create_subscription(LaserScan, "/scan", self._scan, 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(PointStamped, "/rl_goal", self._goal, 10)
        self.cmd = self.create_publisher(Twist, "/cmd_shielded", 10)
        self.wpub = self.create_publisher(Float32MultiArray, "/policy_weights", 10)
        self.create_timer(DT, self._tick)
        self.get_logger().warn(
            "PLACEHOLDER controller running (no trained policy loaded) — "
            "heuristic goal-seeking + real safety shield only")

    def _scan(self, s):
        self.lidar = scan_to_lidar48(s.ranges, s.angle_min, s.angle_increment, s.range_max, self.offset)

    def _odom(self, o):
        pass

    def _goal(self, p):
        self.goal = (p.point.x, p.point.y)

    def _pose(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y, _yaw(t.transform.rotation)
        except Exception:
            return None

    def _tick(self):
        p = self._pose()
        if p is None or self.goal is None:
            return
        x, y, yaw = p
        gx, gy = self.goal
        bearing = math.atan2(gy - y, gx - x) - yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        dist = math.hypot(gx - x, gy - y)
        v_raw = MAX_LINEAR * min(dist / 1.5, 1.0) * max(0.0, 1.0 - abs(bearing) / (math.pi / 2))
        delta_raw = max(-1.0, min(1.0, bearing)) * math.radians(30)
        v, delta = shield(v_raw, delta_raw, self.lidar)
        out = Twist()
        out.linear.x = v
        out.angular.z = delta
        self.cmd.publish(out)
        w = Float32MultiArray()
        w.data = [1.0, 0.0, 0.0, 0.0]
        self.wpub.publish(w)


def main():
    rclpy.init()
    rclpy.spin(PlaceholderController())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
