"""Tier-1 shield node. Filters a raw [v, steering] command against the LiDAR.

Subscribes:  /scan (sensor_msgs/LaserScan), /cmd_raw (geometry_msgs/Twist)
Publishes:   /cmd_shielded (geometry_msgs/Twist)   [linear.x=v m/s, angular.z=steering rad]

Run standalone (Tier 1) with a teleop source on /cmd_raw, or downstream of the
policy (the controller_node publishes /cmd_raw). Loop at 10 Hz (DT).
"""
from __future__ import annotations
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from .scan_utils import scan_to_lidar48, shield, N_RAYS, DT


class ShieldNode(Node):
    def __init__(self):
        super().__init__("safety_shield")
        self.declare_parameter("bearing_offset", 0.0)      # rad, LiDAR 0deg vs robot +x
        self.offset = float(self.get_parameter("bearing_offset").value)
        self.lidar = np.ones(N_RAYS, np.float32)
        self.raw = (0.0, 0.0)                              # latest [v, steering]
        self.create_subscription(LaserScan, "/scan", self._scan, 10)
        self.create_subscription(Twist, "/cmd_raw", self._cmd, 10)
        self.pub = self.create_publisher(Twist, "/cmd_shielded", 10)
        self.create_timer(DT, self._tick)                 # 10 Hz
        self.get_logger().info("safety_shield up (10 Hz)")

    def _scan(self, s: LaserScan):
        self.lidar = scan_to_lidar48(s.ranges, s.angle_min, s.angle_increment,
                                     s.range_max, self.offset)

    def _cmd(self, t: Twist):
        self.raw = (t.linear.x, t.angular.z)

    def _tick(self):
        v, d = shield(self.raw[0], self.raw[1], self.lidar)
        out = Twist(); out.linear.x = v; out.angular.z = d
        self.pub.publish(out)


def main():
    rclpy.init(); rclpy.spin(ShieldNode()); rclpy.shutdown()


if __name__ == "__main__":
    main()
