"""Sends one Nav2 NavigateToPose goal, retrying until it's accepted — stands
in for clicking 'Nav2 Goal' in RViz, so a headless Gazebo run drives itself.
Retries because early attempts routinely get rejected ("out of costmap" /
"start pose is out of costmap") while SLAM's map hasn't grown to cover the
goal yet, or the action server isn't fully up — a single try isn't reliable.
"""
from __future__ import annotations
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose


class Nav2GoalSender(Node):
    def __init__(self):
        super().__init__("nav2_goal_sender")
        self.gx = float(self.declare_parameter("goal_x", 3.5).value)
        self.gy = float(self.declare_parameter("goal_y", -1.0).value)
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.busy = False
        self.done = False
        self.create_timer(1.0, self._try_send)

    def _try_send(self):
        if self.busy or self.done:
            return
        if not self.client.wait_for_server(timeout_sec=0.1):
            self.get_logger().info("waiting for navigate_to_pose action server...")
            return
        self.busy = True
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.gx
        goal.pose.pose.position.y = self.gy
        goal.pose.pose.orientation.w = 1.0
        self.get_logger().info(f"sending Nav2 goal ({self.gx}, {self.gy})")
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Nav2 goal rejected, retrying next tick")
            self.busy = False
            return
        self.get_logger().info("Nav2 goal accepted")
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        status = future.result().status
        self.get_logger().info(f"Nav2 result: {status}")
        if status == 4:  # SUCCEEDED
            self.done = True
        else:
            self.busy = False  # aborted/canceled — retry the same goal


def main():
    rclpy.init()
    rclpy.spin(Nav2GoalSender())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
