"""WSL dry-run stub: publishes FAKE sensor topics so the whole pipeline
(controller -> shield) runs on your laptop with NO robot attached. Use this to
learn ROS 2 and verify the policy loads and produces actions before touching the Pi.

Publishes: /scan (a room-like scan with one obstacle ahead), /odom, /people,
           /rl_goal. Then run controller + watch /cmd_shielded, /policy_weights.
"""
from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class SimStub(Node):
    def __init__(self):
        super().__init__("sim_stub")
        self.t = 0.0
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.ppl_pub = self.create_publisher(Float32MultiArray, "/people", 10)
        self.goal_pub = self.create_publisher(PointStamped, "/rl_goal", 10)
        self.tfb = TransformBroadcaster(self)
        self.create_timer(0.1, self.tick)     # 10 Hz
        self.get_logger().info("sim_stub publishing fake /scan /odom /people /rl_goal")

    def tick(self):
        self.t += 0.1
        now = self.get_clock().now().to_msg()
        # 360-beam scan: 12 m everywhere except a 2 m obstacle straight ahead
        n = 360
        s = LaserScan(); s.header.stamp = now; s.header.frame_id = "laser"
        s.angle_min = -math.pi; s.angle_max = math.pi; s.angle_increment = 2*math.pi/n
        s.range_min = 0.05; s.range_max = 12.0
        r = np.full(n, 12.0, np.float32); r[n//2-8:n//2+8] = 2.0     # obstacle ahead
        s.ranges = r.tolist(); self.scan_pub.publish(s)
        # odom: crawling forward
        o = Odometry(); o.header.stamp = now; o.header.frame_id = "odom"; o.child_frame_id = "base_link"
        o.twist.twist.linear.x = 0.2; self.odom_pub.publish(o)
        # map->base_link TF so the controller's pose lookup works
        tf = TransformStamped(); tf.header.stamp = now; tf.header.frame_id = "map"; tf.child_frame_id = "base_link"
        tf.transform.translation.x = 20.0; tf.transform.translation.y = 15.0
        tf.transform.rotation.w = 1.0; self.tfb.sendTransform(tf)
        # one person 3 m ahead, and a goal 6 m ahead
        m = Float32MultiArray(); m.data = [0.6, 0.0, 0.0, 0.0] + [0.0]*16; self.ppl_pub.publish(m)
        g = PointStamped(); g.header.frame_id = "map"; g.point.x = 26.0; g.point.y = 15.0; self.goal_pub.publish(g)


def main():
    rclpy.init(); rclpy.spin(SimStub()); rclpy.shutdown()


if __name__ == "__main__":
    main()
