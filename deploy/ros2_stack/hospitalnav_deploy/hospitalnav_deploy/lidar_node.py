"""Self-contained LD19 LiDAR node: reads the sensor via our own ld19_driver and
publishes /scan (sensor_msgs/LaserScan). Use this OR the off-the-shelf
ldlidar_stl_ros2 node - not both. Params: port, baud, frame_id.
"""
from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from .ld19_driver import LD19Lidar


class LidarNode(Node):
    def __init__(self):
        super().__init__("ld19_lidar")
        port = self.declare_parameter("port", "/dev/ttyUSB0").value
        baud = int(self.declare_parameter("baud", 230400).value)
        self.frame = self.declare_parameter("frame_id", "laser").value
        self.lidar = LD19Lidar(port, baud); self.lidar.start()
        self.pub = self.create_publisher(LaserScan, "/scan", 10)
        self.create_timer(0.05, self.tick)     # 20 Hz
        self.get_logger().info(f"LD19 lidar on {port} -> /scan")

    def tick(self):
        f = self.lidar.get_latest_scan()
        s = LaserScan()
        s.header.stamp = self.get_clock().now().to_msg(); s.header.frame_id = self.frame
        s.angle_min = 0.0; s.angle_max = 2 * math.pi; s.angle_increment = math.pi / 180.0
        s.range_min = 0.05; s.range_max = 12.0
        r = f.distances_m.astype(float); r[~np.isfinite(r)] = 12.0
        s.ranges = r.tolist()
        self.pub.publish(s)

    def destroy_node(self):
        self.lidar.stop(); super().destroy_node()


def main():
    rclpy.init(); node = LidarNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
