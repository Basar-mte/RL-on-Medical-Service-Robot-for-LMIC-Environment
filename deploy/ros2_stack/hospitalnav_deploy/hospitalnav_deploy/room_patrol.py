"""Room-to-room patrol: cycles Nav2 NavigateToPose goals through the corridor
and all seven rooms, with no human in the loop — stands in for someone
clicking 'Nav2 Goal' in RViz over and over. Starts at the nurse desk
(training/layout.py's START_POINT) and its first real destination is the
emergency room; by default (stop_at_first_goal:=true) it stops there after
that one attempt instead of continuing to tour the rest of the floor. None of
the south rooms connect to each other directly (each has exactly one doorway,
onto the corridor/open area), so a full tour would return to a corridor
waypoint before approaching each next room's own door.
"""
from __future__ import annotations
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

WAYPOINTS = [
    # short hops first: Nav2 rejects any goal outside the SLAM map explored so
    # far, and from a standing start that's only what the first scan saw — so
    # bootstrap down the corridor in reach-able steps before the far rooms.
    # Room centres/doors match training/layout.py's ROOMS. Spawn is at the
    # nurse desk corner (37, 15), far east, so the first legs walk west.
    ("corridor (near nurse desk)", 35.0, 15.0),
    ("corridor (bootstrap 3)", 28.0, 15.0),
    ("corridor (from nurse desk)", 24.0, 15.0),
    ("corridor (mid)", 18.5, 15.0),
    ("corridor (bootstrap 2)", 14.0, 15.0),
    ("corridor (bootstrap 1)", 9.0, 15.0),
    ("corridor (near ER)", 5.0, 14.0),
    ("emergency room", 5.0, 7.5),
    ("corridor (near prayer)", 13.0, 14.0),
    ("prayer room", 13.0, 7.5),
    ("corridor (near ward 3)", 23.5, 14.0),
    ("general ward 3", 23.5, 7.5),
    ("corridor (near ward 2)", 31.0, 14.0),
    ("general ward 2", 31.0, 7.5),
    ("male ward", 7.5, 23.5),
    ("corridor (bootstrap 2)", 14.0, 15.0),
    ("general ward 1", 20.0, 23.5),
    ("corridor (east)", 35.0, 15.0),
    ("female ward", 32.5, 23.5),
]


class RoomPatrol(Node):
    def __init__(self):
        super().__init__("room_patrol")
        self.declare_parameter("stop_at_first_goal", True)
        self.declare_parameter("first_goal_name", "emergency room")
        self.stop_at_first_goal = self.get_parameter("stop_at_first_goal").value
        self.first_goal_name = self.get_parameter("first_goal_name").value

        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.goal_pub = self.create_publisher(PoseStamped, "/patrol_goal", 10)
        self.idx = 0
        self.busy = False
        self.stopped = False
        self.current_name = None
        self.create_timer(1.0, self._tick)

    def _tick(self):
        if self.busy or self.stopped:
            return
        if not self.client.wait_for_server(timeout_sec=0.1):
            self.get_logger().info("waiting for navigate_to_pose action server...")
            return
        name, x, y = WAYPOINTS[self.idx]
        self.current_name = name
        self.busy = True
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0
        self.get_logger().info(f"patrol -> {name} ({x}, {y})")
        self.goal_pub.publish(goal.pose)
        self.client.send_goal_async(goal).add_done_callback(self._on_response)

    def _on_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("patrol goal rejected, retrying next tick")
            self.busy = False
            return
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        status = future.result().status
        self.get_logger().info(f"patrol leg finished: status={status}")
        if self.stop_at_first_goal and self.current_name == self.first_goal_name:
            self.stopped = True
            self.busy = False
            self.get_logger().info(
                f"reached '{self.first_goal_name}' on the first attempt — stopping patrol here")
            return
        self.idx = (self.idx + 1) % len(WAYPOINTS)
        self.busy = False


def main():
    rclpy.init()
    rclpy.spin(RoomPatrol())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
