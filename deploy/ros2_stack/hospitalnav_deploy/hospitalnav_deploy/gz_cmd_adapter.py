"""Converts the controller/shield's [v, steering-angle] into the [v, yaw-rate]
Twist that gz-sim's AckermannSteering plugin (urdf's <topic>cmd_vel</topic>)
expects. Same bicycle-model formula used everywhere else in this project
(scan_utils/rrc_bridge/training env): yaw_rate = (v / wheelbase) * tan(delta).

Subscribes: /cmd_shielded (Twist: linear.x=v m/s, angular.z=steering rad)
Publishes:  /cmd_vel (Twist: linear.x=v m/s, angular.z=yaw rate rad/s)
"""
from __future__ import annotations
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

WHEELBASE = 0.20  # must match urdf/hospitalbot.urdf's <wheel_base>


class GzCmdAdapter(Node):
    def __init__(self):
        super().__init__("gz_cmd_adapter")
        self.create_subscription(Twist, "/cmd_shielded", self._cmd, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _cmd(self, t: Twist):
        v, delta = t.linear.x, t.angular.z
        yaw_rate = (v / WHEELBASE) * math.tan(delta)
        out = Twist()
        out.linear.x = v
        out.angular.z = yaw_rate
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(GzCmdAdapter())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
