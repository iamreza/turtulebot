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

from .config import NUM_SLOTS


class AppState:
    def __init__(self):
        self.lock = threading.RLock()

        # Live camera. update_frame stores only the raw frame (cheap) so the ROS
        # callback never blocks; the mask feed computes its mask on demand.
        self.latest_raw = None
        self.last_frame_time = None
        self.latest_boxes = []          # most recent detection bboxes, for the live overlay

        # Workflow data: scan 1 = references, scan 2 = checks.
        self.references = {i: None for i in range(1, NUM_SLOTS + 1)}        # prepared gray (in-memory)
        self.reference_images = {i: None for i in range(1, NUM_SLOTS + 1)}  # saved crop path
        self.check_images = {i: None for i in range(1, NUM_SLOTS + 1)}      # saved crop path
        self.checks = {i: None for i in range(1, NUM_SLOTS + 1)}            # verdict dict

        # UI / status
        self.status_text = "WAITING_FOR_CAMERA"
        self.result_text = "0/5 detected"
        self.last_detect_summary = "-"
        self.workflow_active = False
        self.ui_phase = "baseline"

        # Odom (set by CameraNode.odom_callback)
        self.current_yaw = None
        self.base_last_odom_time = None

        # Workflow thread control
        self.workflow_thread = None
        self.stop_event = threading.Event()

    # -------- camera --------
    def update_frame(self, frame):
        with self.lock:
            self.latest_raw = frame.copy()
            self.last_frame_time = time.time()
            if self.status_text == "WAITING_FOR_CAMERA":
                self.status_text = "READY - click Start to detect cards"

    def get_frame(self):
        with self.lock:
            return self.latest_raw.copy() if self.latest_raw is not None else None

    def get_frame_with_time(self):
        # Frame plus its arrival timestamp, so callers can tell distinct frames
        # apart (the sharpest-frame sampler skips frames it has already scored).
        with self.lock:
            if self.latest_raw is None:
                return None, None
            return self.latest_raw.copy(), self.last_frame_time

    # -------- workflow setters --------
    def set_phase(self, ui_phase, status_text=None):
        with self.lock:
            self.ui_phase = ui_phase
            if status_text is not None:
                self.status_text = status_text

    def set_boxes(self, boxes):
        with self.lock:
            self.latest_boxes = list(boxes)

    def set_references(self, prepared, image_paths):
        with self.lock:
            for i in range(1, NUM_SLOTS + 1):
                self.references[i] = prepared.get(i)
                self.reference_images[i] = image_paths.get(i)

    def set_checks(self, image_paths, verdicts):
        with self.lock:
            for i in range(1, NUM_SLOTS + 1):
                self.check_images[i] = image_paths.get(i)
                self.checks[i] = verdicts.get(i)

    def reference_count(self):
        with self.lock:
            return sum(1 for i in range(1, NUM_SLOTS + 1) if self.reference_images[i] is not None)

    # -------- clear --------
    def clear(self):
        with self.lock:
            self.latest_boxes = []
            self.references = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.reference_images = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.check_images = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.checks = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.last_detect_summary = "-"
            self.result_text = "0/5 detected"
            self.status_text = "READY - click Start to detect cards"
            self.workflow_active = False
            self.ui_phase = "baseline"

    # -------- UI view --------
    def slot_info(self):
        slots = []
        with self.lock:
            for i in range(1, NUM_SLOTS + 1):
                chk = self.checks[i]
                slots.append({
                    "slot_id": i,
                    "reference_image": self.reference_images[i],
                    "check_image": self.check_images[i],
                    "check_orientation": chk["orientation"] if chk else None,
                    "is_rotated": bool(chk["is_rotated"]) if chk else False,
                })
        return slots
