"""Re-broadcasts the bridged /odom (from gz-sim's AckermannSteering plugin,
via ros_gz_bridge) as an odom->base_link TF. SLAM Toolbox and controller_node's
TF lookups both need this; the bridge only forwards the Odometry message
itself, not gz-sim's internal transform tree, so this makes it a real ROS TF.

AckermannSteering zeroes its odometry AT THE SPAWN POINT, not at Gazebo's
world origin (classic Gazebo's planar_move used absolute world pose instead —
a different convention). Left uncorrected, SLAM's map frame ends up anchored
at wherever the robot happened to spawn, while every layout coordinate
(room_patrol's waypoints, and controller_node's own SemanticMap zone lookup +
robot_state normalization, both keyed on the [0,40]x[0,30] layout with origin
at its bottom-left corner) assumes map-frame == that same absolute layout
frame. Adding the known spawn offset here makes that true again.
"""
from __future__ import annotations
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__("odom_tf_broadcaster")
        self.spawn_x = float(self.declare_parameter("spawn_x", 0.0).value)
        self.spawn_y = float(self.declare_parameter("spawn_y", 0.0).value)
        self.tfb = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/odom", self._odom, 10)

    def _odom(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = msg.pose.pose.position.x + self.spawn_x
        t.transform.translation.y = msg.pose.pose.position.y + self.spawn_y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.tfb.sendTransform(t)


def main():
    rclpy.init()
    rclpy.spin(OdomTfBroadcaster())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
