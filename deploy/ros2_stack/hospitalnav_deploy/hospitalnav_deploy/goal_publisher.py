"""Publishes a fixed goal point (map frame) at 2 Hz. Default topic /rl_goal feeds
the controller directly (mapless policies, Gazebo smoke-test stage); with the
route planner in the loop publish the FINAL destination to /nav_goal instead
(`-p topic:=/nav_goal`) and let route_planner produce /rl_goal.
"""
from __future__ import annotations
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


class GoalPublisher(Node):
    def __init__(self):
        super().__init__("goal_publisher")
        self.gx = float(self.declare_parameter("goal_x", 3.5).value)
        self.gy = float(self.declare_parameter("goal_y", -1.0).value)
        topic = str(self.declare_parameter("topic", "/rl_goal").value)
        self.pub = self.create_publisher(PointStamped, topic, 10)
        self.create_timer(0.5, self._tick)

    def _tick(self):
        m = PointStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.point.x = self.gx
        m.point.y = self.gy
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(GoalPublisher())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
