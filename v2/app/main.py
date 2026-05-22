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

import rclpy
from rclpy.executors import ExternalShutdownException

from .camera_node import CameraNode
from .config import PORT
from .logger import logger
from .state import AppState
from .web import run_flask, set_node, set_state


def main():
    app_state = AppState()
    set_state(app_state)

    rclpy.init()
    node = CameraNode(app_state, logger)
    set_node(node)

    flask_thread = threading.Thread(target=run_flask, name="FlaskThread", daemon=True)
    flask_thread.start()
    logger.log(f"Magic Card v2 started: http://0.0.0.0:{PORT}")

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        logger.log("Magic Card v2 shutting down")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
