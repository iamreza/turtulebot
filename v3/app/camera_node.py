# Course: Autonomous Robotics
# Supervisor: Prof. Dr.-Ing. Reinhard Gerndt
# Semester: Sommer Semester
# Group: 7
#
# Team:
# - Reza Babaee, 70498082
# - Hamid Safisamghabadi, 70497663
# - Emad Mohammadi, 70494663
# - Azarjan Gharibian

import time

import cv2
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image
from std_srvs.srv import SetBool

from .config import (
    BASE_ALIVE_TIMEOUT,
    BATTERY_TOPIC,
    CAMERA_RAW_TOPIC,
    CMD_PUBLISH_PERIOD,
    CMD_VEL_TOPIC,
    ODOM_TOPIC,
)
from .utils import quaternion_to_yaw, shortest_angle_diff


class CameraNode(Node):
    def __init__(self, state, logger):
        super().__init__("magic_card_v2_camera")
        self.state = state
        self.logger = logger
        self.frames_received = 0
        self.battery_percent = None
        self.battery_state = "unknown"
        self.battery_last_time = None

        self.create_subscription(Image, CAMERA_RAW_TOPIC, self.image_callback, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        self.create_subscription(BatteryState, BATTERY_TOPIC, self.battery_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.motor_power_client = self.create_client(SetBool, "/motor_power")

        self.logger.log(f"CameraNode subscribed to {CAMERA_RAW_TOPIC}, {ODOM_TOPIC}, and {BATTERY_TOPIC}")

    # -------- camera --------
    def image_callback(self, msg):
        try:
            frame = self._decode_image(msg)
        except Exception as exc:
            self.logger.log(f"camera decode failed: {exc}")
            return
        self.frames_received += 1
        self.state.update_frame(frame)

    def _decode_image(self, msg):
        enc = (msg.encoding or "").lower()
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if enc in ("bgr8", "8uc3"):
            return arr.reshape((msg.height, msg.width, 3)).copy()
        if enc == "rgb8":
            rgb = arr.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if enc in ("mono8", "8uc1"):
            gray = arr.reshape((msg.height, msg.width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if enc == "bgra8":
            bgra = arr.reshape((msg.height, msg.width, 4))
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        if enc == "rgba8":
            rgba = arr.reshape((msg.height, msg.width, 4))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        raise ValueError(f"unsupported encoding: {msg.encoding}")

    # -------- odom --------
    def odom_callback(self, msg):
        try:
            yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        except Exception:
            yaw = None
        with self.state.lock:
            self.state.base_last_odom_time = time.time()
            if yaw is not None:
                self.state.current_yaw = yaw

    def base_alive(self):
        with self.state.lock:
            last = self.state.base_last_odom_time
        return last is not None and (time.time() - last) <= BASE_ALIVE_TIMEOUT

    # -------- battery --------
    def battery_callback(self, msg):
        percent = None
        try:
            if msg.percentage >= 0:
                value = float(msg.percentage)
                if value <= 1.0:
                    value *= 100.0
                percent = max(0, min(100, round(value)))
        except Exception:
            percent = None

        status = getattr(msg, "power_supply_status", 0)
        if status == BatteryState.POWER_SUPPLY_STATUS_CHARGING:
            state = "charging"
        elif percent is not None and percent <= 15:
            state = "low"
        elif percent is not None and percent <= 35:
            state = "mid"
        elif percent is not None:
            state = "ok"
        else:
            state = "unknown"

        with self.state.lock:
            self.battery_percent = percent
            self.battery_state = state
            self.battery_last_time = time.time()

    def battery_status(self):
        with self.state.lock:
            percent = self.battery_percent
            state = self.battery_state
            last_time = self.battery_last_time
        if last_time is None:
            return {"percent": None, "state": "unknown", "source": "none"}
        return {
            "percent": percent,
            "state": state,
            "age": round(time.time() - last_time, 2),
            "source": BATTERY_TOPIC,
        }

    # -------- cmd_vel --------
    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0)

    def cmd_vel_subscriber_count(self):
        try:
            return int(self.cmd_pub.get_subscription_count())
        except Exception:
            return 0

    def ensure_motor_power(self):
        # TurtleBot3 OpenCR holds motor torque off until /motor_power is set
        # True; without this the wheels never spin and the closed-loop turn
        # times out at rotated=0. Mirrors v1 robot_node.ensure_motor_power.
        if not self.motor_power_client.service_is_ready():
            if not self.motor_power_client.wait_for_service(timeout_sec=0.5):
                return False, "Motor power service not available"

        request = SetBool.Request()
        request.data = True
        future = self.motor_power_client.call_async(request)

        start = time.time()
        while not future.done() and time.time() - start < 2.0:
            time.sleep(0.05)
        if not future.done():
            return False, "Motor power enable timed out"

        try:
            response = future.result()
        except Exception as exc:
            return False, f"Motor power enable failed: {exc}"
        if not response.success:
            return False, f"Motor power rejected: {response.message}"
        return True, "Motor power enabled"

    def turn_by_angle_blocking(self, target_angle_rad, angular_speed, label, stop_event=None):
        # Closed-loop rotation using /odom yaw. Drives at angular_speed in the
        # direction of sign(target_angle_rad) until the robot has rotated by
        # |target_angle_rad|. More accurate than time-based turns (ignores motor
        # stiction / battery variation). Ported from v1 robot_node.
        # /cmd_vel having no subscriber usually means the robot base driver
        # (e.g. turtlebot3_node) is not running. We only warn (discovery can lag,
        # and publishing to nobody is harmless); /odom is the real gate below
        # since the closed-loop turn cannot work without yaw feedback.
        if self.cmd_vel_subscriber_count() <= 0:
            self.logger.log(f"Odom turn warning: {label} | no /cmd_vel subscriber (is the robot base running?)")
        if not self.base_alive():
            self.logger.log(f"Odom turn rejected: {label} | no /odom (robot base not running?)")
            return False, "Robot base not publishing /odom"

        with self.state.lock:
            start_yaw = self.state.current_yaw
        if start_yaw is None:
            self.logger.log(f"Odom turn rejected: {label} | no yaw yet")
            return False, "No yaw available"

        ok, msg = self.ensure_motor_power()
        if not ok:
            self.logger.log(f"Odom turn rejected: {label} | {msg}")
            return False, msg

        target_abs = abs(target_angle_rad)
        direction = 1.0 if target_angle_rad > 0 else -1.0
        angular_z = direction * abs(angular_speed)

        # Stop early to compensate for continued rotation after cmd_vel=0
        # (TurtleBot3 burger keeps rotating ~7-10 deg after stop).
        stop_margin_rad = 0.13   # ~7.5 deg
        expected_t = target_abs / max(abs(angular_speed), 0.01)
        timeout_s = max(2.0, expected_t * 3.0)

        self.logger.log(
            f"Odom turn start: {label} | target={target_angle_rad:+.2f}rad | "
            f"speed={abs(angular_speed):.2f} | start_yaw={start_yaw:+.2f}"
        )

        rotated = 0.0
        last_yaw = start_yaw
        start_time = time.time()
        ok = True
        fail_reason = None

        while abs(rotated) < (target_abs - stop_margin_rad):
            if stop_event is not None and stop_event.is_set():
                ok, fail_reason = False, "stopped"
                break
            if not self.base_alive():
                ok, fail_reason = False, "base_died"
                break
            if time.time() - start_time > timeout_s:
                ok, fail_reason = False, "timeout"
                break

            with self.state.lock:
                cur_yaw = self.state.current_yaw
            if cur_yaw is None:
                time.sleep(CMD_PUBLISH_PERIOD)
                continue

            rotated += shortest_angle_diff(cur_yaw, last_yaw)
            last_yaw = cur_yaw

            self.publish_cmd(0.0, angular_z)
            time.sleep(CMD_PUBLISH_PERIOD)

        self.publish_stop()
        time.sleep(0.15)

        if not ok:
            self.logger.log(f"Odom turn end: {label} | FAILED ({fail_reason}) | rotated={rotated:+.2f}rad")
            return False, fail_reason or "turn_failed"

        self.logger.log(f"Odom turn end: {label} | rotated={rotated:+.2f}rad (target={target_angle_rad:+.2f})")
        return True, f"rotated {rotated:+.2f}rad"
