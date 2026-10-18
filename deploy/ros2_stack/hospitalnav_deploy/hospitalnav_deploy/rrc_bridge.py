"""RRC-lite hardware bridge, wired to Yahboom's real Rosmaster_Lib SDK.

API reference: https://www.yahboom.net (Rosmaster driver library install page) —
Rosmaster_Lib is NOT on GitHub, only distributed as a py_install.zip bundle;
install it on the Pi with `sudo python3 setup.py install` from that bundle,
then `from Rosmaster_Lib import Rosmaster` works.

This chassis is genuine Ackermann, not skid-steer: two rear wheels are driven
(always at the SAME commanded speed — there's no differential/skid steering),
and the front two wheels share ONE steering-servo mechanism via a linkage. So
instead of the higher-level, car-type-dependent set_car_motion(), this uses
the two unambiguous, documented primitives directly:
  - set_motor(speed_1, speed_2, speed_3, speed_4)  — PWM speed per channel, [-100, 100]
  - set_pwm_servo(servo_id, angle)                  — one servo, angle in [0, 180]
Heading is integrated with the standard bicycle model (v, steering-angle ->
yaw rate) since there's no wheel-speed differential to read yaw from; only
the linear speed comes from the rear encoders.

Subscribes: /cmd_shielded (Twist)  [linear.x = v m/s, angular.z = steering rad]
Publishes:  /odom (Odometry); broadcasts odom->base_link TF.

Everything below marked TODO needs a bench check (wheels off the ground)
before trusting it on the floor — these are physical wiring/calibration
facts about YOUR specific build that no generic API doc can supply:
  - REAR_MOTOR_CHANNELS: which of set_motor's 4 channels the rear wheels are on
  - SPEED_TO_PWM, WHEEL_R, TICKS_PER_REV: real-world speed <-> PWM/encoder scale
  - STEER_SERVO_ID, STEER_CENTER_DEG, STEER_DEG_PER_RAD: which servo channel,
    its straight-ahead angle, and its degrees-of-travel per radian of steering
"""
from __future__ import annotations
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

MAX_LINEAR, MAX_STEER = 0.8, math.radians(30)
WHEELBASE = 0.20        # must match sim
WATCHDOG = 0.3          # s; zero output if no command within this

# --- drive: two rear motors, always the same speed (Ackermann, no skid-steer) ---
REAR_MOTOR_CHANNELS = (1, 2)          # TODO verify: set_motor channel indices for the rear wheels
SPEED_TO_PWM = 100.0 / MAX_LINEAR      # TODO tune: PWM% per m/s — measure real speed at a known PWM
WHEEL_R = 0.05                         # TODO: drive wheel radius (m)
TICKS_PER_REV = 660                    # TODO: encoder ticks per wheel revolution (motor/encoder datasheet)

# --- steering: one servo, shared linkage to both front wheels ---
STEER_SERVO_ID = 1                     # TODO verify: set_pwm_servo channel for the steering servo
STEER_CENTER_DEG = 90.0                # TODO calibrate: servo angle that points the wheels straight ahead
STEER_DEG_PER_RAD = 60.0 / MAX_STEER   # TODO calibrate: servo degrees of travel per radian of our steering


class RRCBridge(Node):
    def __init__(self):
        super().__init__("rrc_bridge")
        self.declare_parameter("port", "/dev/ttyUSB0")
        port = self.get_parameter("port").value

        from Rosmaster_Lib import Rosmaster
        # car_type mostly matters for the higher-level set_car_motion() kinematics;
        # since we drive the motor/servo channels directly below, its exact value
        # is less critical here, but 1 (Yahboom's default) is a safe starting point.
        self.bot = Rosmaster(car_type=1, com=port, delay=0.002, debug=False)
        self.bot.create_receive_threading()   # starts the MCU data-read thread

        self.last_cmd = time.time()
        self.x = self.y = self.th = 0.0
        self.last_t = time.time()
        self.last_steer = 0.0
        self.prev_encoders = None
        self.create_subscription(Twist, "/cmd_shielded", self._cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tfb = TransformBroadcaster(self)
        self.create_timer(0.02, self._loop)     # 50 Hz odom + watchdog
        self.get_logger().info(f"rrc_bridge up: Rosmaster on {port}")

    def _cmd(self, t: Twist):
        v = max(-MAX_LINEAR, min(MAX_LINEAR, t.linear.x))
        steer = max(-MAX_STEER, min(MAX_STEER, t.angular.z))
        self.last_cmd = time.time()
        self._drive(v, steer)

    def _drive(self, v, steer):
        self.last_steer = steer
        pwm = v * SPEED_TO_PWM
        speeds = [0.0, 0.0, 0.0, 0.0]
        for ch in REAR_MOTOR_CHANNELS:
            speeds[ch - 1] = pwm
        self.bot.set_motor(*speeds)
        servo_deg = max(0.0, min(180.0, STEER_CENTER_DEG + steer * STEER_DEG_PER_RAD))
        self.bot.set_pwm_servo(STEER_SERVO_ID, servo_deg)

    def _loop(self):
        now = time.time()
        if now - self.last_cmd > WATCHDOG:               # watchdog: stop on stale cmd
            self._drive(0.0, 0.0)

        # --- odometry: real speed from the rear encoders, heading from the
        # bicycle model — steering is a separate actuator here, not a
        # wheel-speed differential, so there's no left/right yaw signal ---
        encoders = self.bot.get_motor_encoder()
        dt = now - self.last_t; self.last_t = now
        v = 0.0
        if self.prev_encoders is not None and dt > 0:
            ticks = [encoders[ch - 1] - self.prev_encoders[ch - 1] for ch in REAR_MOTOR_CHANNELS]
            avg_ticks = sum(ticks) / len(ticks)
            wheel_rev_per_s = (avg_ticks / TICKS_PER_REV) / dt
            v = wheel_rev_per_s * 2 * math.pi * WHEEL_R
        self.prev_encoders = encoders

        self.th += (v / WHEELBASE) * math.tan(self.last_steer) * dt
        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        o = Odometry(); o.header.stamp = self.get_clock().now().to_msg()
        o.header.frame_id = "odom"; o.child_frame_id = "base_link"
        o.pose.pose.position.x = self.x; o.pose.pose.position.y = self.y
        o.pose.pose.orientation.z = math.sin(self.th / 2); o.pose.pose.orientation.w = math.cos(self.th / 2)
        o.twist.twist.linear.x = v
        self.odom_pub.publish(o)
        tf = TransformStamped(); tf.header = o.header; tf.child_frame_id = "base_link"
        tf.transform.translation.x = self.x; tf.transform.translation.y = self.y
        tf.transform.rotation = o.pose.pose.orientation
        self.tfb.sendTransform(tf)

    def destroy_node(self):
        self._drive(0.0, 0.0)
        super().destroy_node()


def main():
    rclpy.init(); rclpy.spin(RRCBridge()); rclpy.shutdown()


if __name__ == "__main__":
    main()
