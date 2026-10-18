"""Records the numbers from a sim run to a CSV for later inspection/plotting.

Subscribes: /cmd_shielded (Twist), /odom (Odometry), /rl_goal (PointStamped),
            TF map->base_link
Writes one row every tick (DT) to the path in the 'csv_path' parameter:
  t, v_cmd, steering_cmd, x, y, yaw, goal_x, goal_y, goal_dist
"""
from __future__ import annotations
import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped
from tf2_ros import Buffer, TransformListener
from .scan_utils import DT


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class CsvLogger(Node):
    def __init__(self):
        super().__init__("hospitalnav_csv_logger")
        path = self.declare_parameter("csv_path", "/tmp/hospitalnav_run.csv").value
        self.f = open(path, "w")
        self.f.write("t,v_cmd,steering_cmd,x,y,yaw,goal_x,goal_y,goal_dist\n")
        self.t0 = time.time()
        self.cmd = (0.0, 0.0)
        self.goal = None
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_subscription(Twist, "/cmd_shielded", self._cmd, 10)
        self.create_subscription(PointStamped, "/rl_goal", self._goal, 10)
        self.create_timer(DT, self._tick)
        self.get_logger().info(f"logging run numbers to {path}")

    def _cmd(self, t: Twist):
        self.cmd = (t.linear.x, t.angular.z)

    def _goal(self, p: PointStamped):
        self.goal = (p.point.x, p.point.y)

    def _tick(self):
        try:
            tr = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            x = tr.transform.translation.x
            y = tr.transform.translation.y
            yaw = _yaw(tr.transform.rotation)
        except Exception:
            return
        gx, gy = self.goal if self.goal else (float("nan"), float("nan"))
        dist = math.hypot(gx - x, gy - y) if self.goal else float("nan")
        t = time.time() - self.t0
        self.f.write(f"{t:.2f},{self.cmd[0]:.4f},{self.cmd[1]:.4f},{x:.3f},{y:.3f},{yaw:.3f},{gx:.3f},{gy:.3f},{dist:.3f}\n")
        self.f.flush()

    def destroy_node(self):
        self.f.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = CsvLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
