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

import threading
import time

import rclpy

from .config import PORT
from .robot_node import RobotAssistedDebugNode
from .utils import ensure_dirs, setup_session_logging
from .web import app, set_robot_node


def main():
    ensure_dirs()
    setup_session_logging()

    rclpy.init()
    robot_node = RobotAssistedDebugNode()
    set_robot_node(robot_node)

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(robot_node,),
        daemon=True
    )
    spin_thread.start()

    try:
        app.run(
            host="0.0.0.0",
            port=PORT,
            debug=False,
            threaded=True
        )

    except KeyboardInterrupt:
        pass

    finally:
        try:
            robot_node.publish_stop()
            time.sleep(0.1)
            robot_node.publish_stop()
            robot_node.add_event("Program shutting down")
        except Exception:
            pass

        try:
            robot_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
