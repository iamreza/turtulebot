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

import math
import os
import threading
import time
import traceback

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, CompressedImage, Image
from std_srvs.srv import SetBool

from .config import *
from .utils import *

# ============================================================
# ROS NODE
# ============================================================

class RobotAssistedDebugNode(Node):
    def __init__(self):
        super().__init__("five_slot_robot_assisted_debug_node")

        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.motor_power_client = self.create_client(SetBool, "/motor_power")

        self.create_subscription(
            CompressedImage,
            CAMERA_COMPRESSED_TOPIC,
            self.compressed_image_callback,
            10
        )

        self.create_subscription(
            Image,
            CAMERA_RAW_TOPIC,
            self.raw_image_callback,
            10
        )

        self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.odom_callback,
            10
        )

        self.create_subscription(
            BatteryState,
            BATTERY_TOPIC,
            self.battery_callback,
            10
        )

        self.lock = threading.RLock()

        self.base_last_odom_time = None
        self.base_odom_count = 0
        self.current_yaw = None
        self.current_pos_x = None  # meters in odom frame
        self.current_pos_y = None
        self.battery_percent = None
        self.battery_state = "unknown"
        self.battery_last_time = None

        self.latest_raw = None
        self.latest_bbox = None
        self.latest_mask = None
        self.latest_crop_raw = None
        self.latest_crop_prepared = None
        self.latest_card_found = False

        self.status_text = "WAITING_FOR_CAMERA"
        self.result_text = "No rotated card detected yet."
        self.last_report_path = "-"

        self.references = {i: None for i in range(1, NUM_SLOTS + 1)}
        self.reference_times = {i: None for i in range(1, NUM_SLOTS + 1)}
        self.reference_images = {i: None for i in range(1, NUM_SLOTS + 1)}  # file paths

        self.checks = {i: None for i in range(1, NUM_SLOTS + 1)}
        self.check_times = {i: None for i in range(1, NUM_SLOTS + 1)}

        self.event_log = []

        self.motion_active = False
        self.stop_motion_requested = False
        self.workflow_active = False
        self.workflow_stop_requested = False

        self.frame_count_compressed = 0
        self.frame_count_raw = 0
        self.last_frame_time = None
        self.last_base_alive_state = None

        # Manual scan phase machine (movement disabled mode):
        #   SAVE_PHASE -> user clicks Scan once per slot to save references
        #   WAIT_PHASE -> 15s countdown after slot 5 saved, user flips a card
        #   CHECK_PHASE -> user clicks Scan once per slot to compare against ref
        #   DONE_PHASE -> result available, click Clear to reset
        self.scan_phase = "SAVE_PHASE"
        self.ui_phase = "idle"
        self.two_step_active = False
        self.two_step_last_view = "-"
        self.one_shot_active = False
        self.one_shot_last_view = "-"
        self.wait_thread = None
        self.wait_started_at = None
        self.cross_match_result = None

        # Per-slot odom (x,y) recorded at save time during AUTO_RUN. Lets
        # the check phase use the ACTUAL measured slot-to-slot distance
        # instead of a fixed SLOT_FORWARD_DISTANCE_M, so variable user
        # spacing is handled automatically.
        self.slot_save_positions = {i: None for i in range(1, NUM_SLOTS + 1)}

        # The "facing the cards" direction, captured at slot 1 of AUTO_RUN
        # right after vision_align settles. After every step transition we
        # snap robot heading back to this canonical yaw so per-step turn
        # errors don't accumulate (otherwise +6° errors over 4 transitions
        # become +20°+ drift, making cards appear at perspective angle).
        self.canonical_yaw = None

        self.trace_lock = threading.Lock()

        self.health_watchdog_thread = threading.Thread(
            target=self.health_watchdog_loop,
            name="HealthWatchdog",
            daemon=True
        )
        self.health_watchdog_thread.start()

        self.trace("INIT", f"Trace log: {TRACE_LOG_FILE}")
        self.trace("INIT", f"Live log: {LIVE_LOG_FILE}")
        self.trace("INIT", f"Camera raw: {CAMERA_RAW_TOPIC} | compressed: {CAMERA_COMPRESSED_TOPIC}")
        self.trace("INIT", f"Cmd_vel: {CMD_VEL_TOPIC} | Odom: {ODOM_TOPIC} | Battery: {BATTERY_TOPIC} | Port: {PORT}")

        self.add_event("Program started")
        self.get_logger().info("Five Slot Robot Assisted Debug started")
        self.get_logger().info(f"Camera raw topic: {CAMERA_RAW_TOPIC}")
        self.get_logger().info(f"Camera compressed topic: {CAMERA_COMPRESSED_TOPIC}")
        self.get_logger().info(f"Cmd topic: {CMD_VEL_TOPIC}")
        self.get_logger().info(f"Web: http://0.0.0.0:{PORT}")

    def set_ui_phase(self, phase):
        with self.lock:
            self.ui_phase = phase

    # ========================================================
    # LOGGING
    # ========================================================

    def add_event(self, text):
        ensure_dirs()

        entry = f"{now_str()} | {text}"

        with self.lock:
            self.event_log.append(entry)

        print("[EVENT]", entry, flush=True)

        try:
            with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception as e:
            print("[LIVE LOG ERROR]", e, flush=True)

        self.trace("EVENT", text)

    def trace(self, category, text):
        ensure_dirs()

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        thread_name = threading.current_thread().name
        line = f"{ts} | {category:<10} | {thread_name:<20} | {text}"

        try:
            with self.trace_lock:
                with open(TRACE_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            print("[TRACE LOG ERROR]", e, flush=True)

    def trace_exc(self, category, where):
        tb = traceback.format_exc()
        for line in tb.splitlines():
            self.trace(category, f"{where} | {line}")

    def health_watchdog_loop(self):
        last_logged_compressed = 0
        last_logged_raw = 0
        last_periodic = time.time()

        while True:
            try:
                time.sleep(1.0)

                with self.lock:
                    last_odom = self.base_last_odom_time
                    prev_state = self.last_base_alive_state
                    fc_c = self.frame_count_compressed
                    fc_r = self.frame_count_raw
                    last_frame = self.last_frame_time
                    current_status = self.status_text
                    cmd_subs = 0

                try:
                    cmd_subs = self.cmd_vel_subscriber_count()
                except Exception:
                    pass

                now = time.time()
                alive_now = (last_odom is not None) and ((now - last_odom) <= BASE_ALIVE_TIMEOUT)

                if prev_state is True and not alive_now:
                    age = (now - last_odom) if last_odom else float("inf")
                    self.trace("BASE", f"Base became DEAD: no odom for {age:.2f}s | cmd_vel_subs={cmd_subs}")
                    with self.lock:
                        self.last_base_alive_state = False

                # Camera freeze detection: had frames before, now none for too long
                if last_frame is not None:
                    frame_age = now - last_frame
                    if (
                        frame_age > CAMERA_FROZEN_TIMEOUT
                        and current_status != "CAMERA_FROZEN"
                        and not current_status.startswith("WORKFLOW_")
                        and current_status not in ("STOP_REQUESTED",)
                    ):
                        with self.lock:
                            self.status_text = "CAMERA_FROZEN"
                        self.trace(
                            "CAMERA",
                            f"Camera FROZEN: no frame for {frame_age:.1f}s "
                            f"(restart 'ros2 run camera_ros camera_node ...')"
                        )
                        self.add_event(f"Camera FROZEN: no frame for {frame_age:.1f}s")

                # periodic heartbeat every 10s
                if now - last_periodic >= 10.0:
                    delta_c = fc_c - last_logged_compressed
                    delta_r = fc_r - last_logged_raw
                    self.trace(
                        "HEARTBEAT",
                        f"base={self.base_health_text()} | cmd_vel_subs={cmd_subs} | "
                        f"frames_compressed_10s={delta_c} | frames_raw_10s={delta_r} | "
                        f"workflow_active={self.workflow_active} | motion_active={self.motion_active}"
                    )
                    last_logged_compressed = fc_c
                    last_logged_raw = fc_r
                    last_periodic = now

            except Exception:
                try:
                    self.trace_exc("WATCHDOG_ERR", "health_watchdog_loop")
                except Exception:
                    pass
                time.sleep(2.0)

    # ========================================================
    # CAMERA
    # ========================================================

    def compressed_image_callback(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        with self.lock:
            self.frame_count_compressed += 1

        self.process_camera_frame(frame)

    def raw_image_callback(self, msg):
        try:
            encoding = msg.encoding.lower()
            arr = np.frombuffer(msg.data, np.uint8)

            if encoding in ("bgr8", "bgr888"):
                frame = arr.reshape((msg.height, msg.width, 3)).copy()
            elif encoding in ("rgb8", "rgb888"):
                rgb = arr.reshape((msg.height, msg.width, 3))
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif encoding == "mono8":
                gray = arr.reshape((msg.height, msg.width))
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif encoding == "nv21":
                yuv = arr.reshape((msg.height * 3 // 2, msg.width))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
            else:
                self.add_event(f"Unsupported raw image encoding: {msg.encoding}")
                return

        except Exception as e:
            self.add_event(f"Raw image decode error: {e}")
            self.trace_exc("CAMERA_ERR", "raw_image_callback decode")
            return

        with self.lock:
            self.frame_count_raw += 1

        self.process_camera_frame(frame)

    def process_camera_frame(self, frame):
        bbox, mask = self.detect_best_card(frame)

        crop_raw = None
        crop_prepared = None
        found = False

        if bbox is not None:
            x, y, w, h = bbox
            crop_raw = frame[y:y + h, x:x + w].copy()

            if crop_raw is not None and crop_raw.size > 0:
                crop_prepared = prepare_crop(crop_raw)

                if crop_prepared is not None:
                    found = True

        with self.lock:
            self.latest_raw = frame.copy()
            self.latest_bbox = bbox
            self.latest_mask = mask.copy() if mask is not None else None
            self.latest_crop_raw = crop_raw.copy() if crop_raw is not None else None
            self.latest_crop_prepared = crop_prepared.copy() if crop_prepared is not None else None
            self.latest_card_found = found
            self.last_frame_time = time.time()

            if self.status_text in ("WAITING_FOR_CAMERA", "CAMERA_FROZEN"):
                self.status_text = self._phase_status_text()

    def detect_best_card(self, frame):
        # Adaptive brightness detector for white cards on dark backgrounds.
        # Uses Otsu to auto-pick a brightness threshold that separates the
        # card (brightest object) from background, regardless of absolute lighting.
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        # Otsu threshold finds the optimal split between dark and bright pixels.
        # Result is the threshold value picked.
        otsu_thr, _ = cv2.threshold(
            v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Use Otsu but never go below absolute floor
        threshold = max(int(otsu_thr) - CARD_BRIGHTNESS_OTSU_MARGIN, CARD_BRIGHTNESS_MIN)

        bright_mask = cv2.inRange(v_channel, threshold, 255)

        # Lenient saturation cap (rejects very colorful objects like pure yellow wood
        # under harsh light, but allows cream/warm-tinted white cards)
        low_sat_mask = cv2.inRange(s_channel, 0, CARD_SATURATION_MAX)
        card_mask = cv2.bitwise_and(bright_mask, low_sat_mask)

        # Keep search inside the vertical region where cards live
        top_y = int(h * CARD_REGION_TOP_FRAC)
        bot_y = int(h * CARD_REGION_BOTTOM_FRAC)
        if top_y > 0:
            card_mask[:top_y, :] = 0
        if bot_y < h:
            card_mask[bot_y:, :] = 0

        # Two-pass morphology to fill card body without merging with pedestal:
        # 1) Small open: remove tiny noise
        # 2) Medium close: fill gaps between card symbols (card body becomes solid)
        # We deliberately AVOID very large kernels that would bridge card->pedestal.
        # Smaller close kernel + single iteration so the card contour does
        # NOT bridge to the wooden pedestal underneath. Symbols inside a
        # white card are only a few pixels wide, so 5x5 single-pass close is
        # plenty to fill them without spanning the gap to the wood.
        small_k = np.ones((3, 3), np.uint8)
        med_k = np.ones((5, 5), np.uint8)
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, small_k)
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, med_k, iterations=1)

        contours, _ = cv2.findContours(
            card_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        cx_frame = w / 2.0
        cy_frame = h / 2.0
        max_w = int(w * CARD_MAX_WIDTH_FRAC)
        max_h = int(h * CARD_MAX_HEIGHT_FRAC)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CARD_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # One-shot full-row cards are smaller in frame than two-step crops.
            if bw < 25 or bh < 45:
                continue
            if bw > max_w or bh > max_h:
                continue

            aspect = bw / float(bh)
            if aspect < CARD_MIN_ASPECT or aspect > CARD_MAX_ASPECT:
                continue

            # Solidity (fraction of bbox filled by contour). Card outline tends
            # to fill its bbox fairly well; a wide pedestal viewed at an angle
            # often has a more irregular fill.
            solidity = area / float(max(bw * bh, 1))

            cx = x + bw / 2.0
            cy = y + bh / 2.0

            # Score components
            x_dist_score = 1.0 - min(1.0, abs(cx - cx_frame) / (w / 2.0))
            area_score = min(1.0, area / float(w * h * 0.15))  # cap at ~15% of frame
            aspect_score = 1.0 - abs(aspect - 0.65) / 0.65     # ideal card aspect ~0.65
            solidity_score = solidity                          # 0..1

            # Pedestals are at the BOTTOM of the frame; cards are above.
            # Reward candidates whose vertical center is in the upper 2/3.
            upper_score = max(0.0, 1.0 - (cy / h))
            position_weight = 1.5 if CARD_PEDESTAL_PENALTY else 0.0

            score = (
                (2.5 * x_dist_score) +
                (1.5 * aspect_score) +
                (1.2 * area_score) +
                (1.0 * solidity_score) +
                (position_weight * upper_score)
            )

            candidates.append({
                "score": score,
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "area": area,
                "aspect": aspect,
                "solidity": solidity,
            })

        if not candidates:
            return None, card_mask

        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]

        margin = 8
        x1 = max(0, best["x"] - margin)
        y1 = max(0, best["y"] - margin)
        x2 = min(w, best["x"] + best["w"] + margin)
        y2 = min(h, best["y"] + best["h"] + margin)

        bbox = (x1, y1, x2 - x1, y2 - y1)
        return bbox, card_mask

    def _bbox_iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter) / float(max(union, 1))

    def _two_step_card_quality(self, frame, bbox):
        # Quality score for robust duplicate handling and future automatic
        # movement. A crop is better when it is complete, not near image
        # borders, centered enough, card-sized, and has a plausible card shape.
        # Returns a dict with total score in [0,1] plus diagnostics.
        if frame is None or bbox is None:
            return {"quality": 0.0, "reason": "no_frame_or_bbox"}

        h, w = frame.shape[:2]
        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0 or w <= 0 or h <= 0:
            return {"quality": 0.0, "reason": "bad_bbox"}

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        aspect = bw / float(max(bh, 1))
        area_ratio = (bw * bh) / float(max(w * h, 1))

        # Border safety: 1.0 means far from image edge, 0.0 means touching edge.
        left = x / float(w)
        right = (w - (x + bw)) / float(w)
        top = y / float(h)
        bottom = (h - (y + bh)) / float(h)
        min_edge = min(left, right, top, bottom)
        edge_score = max(0.0, min(1.0, min_edge / 0.08))

        # Horizontal center is useful, but should not punish side cards too much.
        x_center_score = 1.0 - min(1.0, abs(cx - w / 2.0) / (w / 2.0))
        x_center_score = 0.35 + 0.65 * x_center_score

        # Prefer card-like aspect. Marked cards in this setup are near square-ish
        # portrait after perspective; very narrow vertical edges should lose here.
        aspect_score = max(0.0, 1.0 - abs(aspect - 0.72) / 0.55)

        # Prefer useful size, but avoid forcing a single exact distance.
        if area_ratio < 0.015:
            size_score = area_ratio / 0.015
        elif area_ratio > 0.22:
            size_score = max(0.0, 1.0 - (area_ratio - 0.22) / 0.18)
        else:
            size_score = 1.0

        # White-card body sanity check from the crop itself.
        crop = frame[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        white_ratio = 0.0
        if crop is not None and crop.size > 0:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            s = hsv[:, :, 1]
            white = (v > 135) & (s < 150)
            white_ratio = float(np.mean(white))
        white_score = max(0.0, min(1.0, white_ratio / 0.35))

        quality = (
            0.28 * edge_score
            + 0.18 * x_center_score
            + 0.20 * aspect_score
            + 0.18 * size_score
            + 0.16 * white_score
        )
        quality = float(max(0.0, min(1.0, quality)))
        return {
            "quality": quality,
            "edge_score": float(edge_score),
            "center_score": float(x_center_score),
            "aspect_score": float(aspect_score),
            "size_score": float(size_score),
            "white_score": float(white_score),
            "aspect": float(aspect),
            "area_ratio": float(area_ratio),
            "min_edge": float(min_edge),
        }

    def detect_all_cards(
        self,
        frame,
        max_cards=TWO_STEP_MAX_CARDS_PER_VIEW,
        edge_reject_frac=TWO_STEP_EDGE_REJECT_FRAC,
        min_quality=0.42,
        hard_mask_edge_reject=True,
        allow_edge_touch=False,
    ):
        # v3 multi-card detector. It still uses the proven white-card mask, but
        # adds: stronger border rejection, non-maximum suppression, and a quality
        # score that rejects wall/edge false positives before they reach matching.
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        otsu_thr, _ = cv2.threshold(
            v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = max(int(otsu_thr) - CARD_BRIGHTNESS_OTSU_MARGIN, CARD_BRIGHTNESS_MIN)

        bright_mask = cv2.inRange(v_channel, threshold, 255)
        low_sat_mask = cv2.inRange(s_channel, 0, CARD_SATURATION_MAX)
        card_mask = cv2.bitwise_and(bright_mask, low_sat_mask)

        top_y = int(h * CARD_REGION_TOP_FRAC)
        bot_y = int(h * CARD_REGION_BOTTOM_FRAC)
        if top_y > 0:
            card_mask[:top_y, :] = 0
        if bot_y < h:
            card_mask[bot_y:, :] = 0

        # Border policy is mode-dependent:
        # - Two-step: strict, because edge objects caused false detections.
        # - One-shot: permissive, because the real first/last cards can sit near
        #   the frame edges when all five cards are visible at once.
        edge_x = int(w * float(edge_reject_frac))
        edge_y = int(h * float(edge_reject_frac))
        if hard_mask_edge_reject:
            if edge_x > 0:
                card_mask[:, :edge_x] = 0
                card_mask[:, w - edge_x:] = 0
            if edge_y > 0:
                card_mask[:edge_y, :] = 0
                card_mask[h - edge_y:, :] = 0

        small_k = np.ones((3, 3), np.uint8)
        med_k = np.ones((5, 5), np.uint8)
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, small_k)
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, med_k, iterations=1)

        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        max_w = int(w * CARD_MAX_WIDTH_FRAC)
        max_h = int(h * CARD_MAX_HEIGHT_FRAC)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CARD_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # One-shot full-row cards are smaller in frame than two-step crops.
            if bw < 25 or bh < 45:
                continue
            if bw > max_w or bh > max_h:
                continue

            # Reject border-touching false positives only in strict mode.
            # In one-shot mode, side cards may be close to the frame edge.
            if not allow_edge_touch:
                if x <= edge_x or (x + bw) >= (w - edge_x):
                    continue
                if y <= edge_y or (y + bh) >= (h - edge_y):
                    continue

            aspect = bw / float(max(bh, 1))
            if aspect < max(CARD_MIN_ASPECT, 0.36) or aspect > min(CARD_MAX_ASPECT, 1.05):
                continue
            solidity = area / float(max(bw * bh, 1))
            if solidity < 0.18:
                continue

            margin = 8
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + bw + margin)
            y2 = min(h, y + bh + margin)
            bbox = (x1, y1, x2 - x1, y2 - y1)

            q = self._two_step_card_quality(frame, bbox)
            if q["quality"] < float(min_quality):
                self.trace(
                    "2STEP_DET",
                    f"reject low quality bbox={bbox} q={q['quality']:.3f} "
                    f"aspect={q.get('aspect', 0):.2f} edge={q.get('edge_score', 0):.2f} "
                    f"white={q.get('white_score', 0):.2f}"
                )
                continue

            # Detector score favours real, complete cards; quality dominates over
            # raw area so large edge noise cannot win.
            cy = y + bh / 2.0
            upper_score = max(0.0, 1.0 - (cy / h))
            area_score = min(1.0, area / float(w * h * 0.10))
            aspect_score = max(0.0, 1.0 - abs(aspect - 0.65) / 0.65)
            det_score = (
                1.0 * area_score
                + 1.0 * aspect_score
                + 0.6 * solidity
                + 0.5 * upper_score
                + 2.2 * q["quality"]
            )

            candidates.append({
                "bbox": bbox,
                "score": float(det_score),
                "area": float(area),
                "quality": q,
            })

        # Non-maximum suppression: keep strongest candidate when contours overlap.
        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = []
        for cand in candidates:
            if any(self._bbox_iou(cand["bbox"], old["bbox"]) > TWO_STEP_NMS_IOU_THRESHOLD for old in selected):
                continue
            selected.append(cand)
            if len(selected) >= max_cards:
                break

        # Sort spatially left->right for baseline slot assignment. Flexible check
        # does not rely on this order, but the overview remains readable.
        selected.sort(key=lambda c: c["bbox"][0] + c["bbox"][2] / 2.0)
        self.trace(
            "2STEP_DET",
            "selected=" + str([
                {"bbox": c["bbox"], "score": round(c["score"], 3), "q": round(c["quality"]["quality"], 3)}
                for c in selected
            ])
        )
        return [c["bbox"] for c in selected], card_mask

    # ========================================================
    # DISPLAY
    # ========================================================

    def build_display_frame(self):
        with self.lock:
            raw = self.latest_raw.copy() if self.latest_raw is not None else None
            bbox = self.latest_bbox
            mask = self.latest_mask.copy() if self.latest_mask is not None else None
            status_text = self.status_text
            result_text = self.result_text
            report_path = self.last_report_path
            workflow_active = self.workflow_active
            motion_active = self.motion_active
            cmd_vel_subscribers = self.cmd_vel_subscriber_count()

            refs_saved = {
                i: self.references[i] is not None
                for i in range(1, NUM_SLOTS + 1)
            }

            checks_text = {}
            for i in range(1, NUM_SLOTS + 1):
                if self.checks[i] is None:
                    checks_text[i] = "-"
                else:
                    c = self.checks[i]
                    checks_text[i] = (
                        f"{c['status']} "
                        f"N={c['score_normal']:.2f} "
                        f"R={c['score_rot180']:.2f}"
                    )

            card_found = self.latest_card_found

        if raw is None:
            left = np.zeros((480, 640, 3), dtype=np.uint8)
            right = np.zeros((480, 640, 3), dtype=np.uint8)
            put_lines(left, ["WAITING FOR CAMERA"], 40, 80, 40, 1.0, (0, 0, 255))
            put_lines(right, ["NO MASK"], 40, 80, 40, 1.0, (255, 255, 255))
            return np.hstack([left, right])

        left = raw.copy()

        h, w = left.shape[:2]
        cx = w // 2

        cv2.line(left, (cx, 0), (cx, h), (255, 0, 0), 3)
        cv2.line(left, (cx - 60, 0), (cx - 60, h), (255, 255, 0), 1)
        cv2.line(left, (cx + 60, 0), (cx + 60, h), (255, 255, 0), 1)

        if bbox is not None:
            x, y, bw, bh = bbox
            cv2.rectangle(left, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            cv2.circle(left, (x + bw // 2, y + bh // 2), 6, (0, 0, 255), -1)
            bbox_line = f"card_bbox: x={x} y={y} w={bw} h={bh}"
        else:
            bbox_line = "card_bbox: NONE"

        lines = [
            f"status: {status_text}",
            f"card_found: {card_found}",
            bbox_line,
            "",
            f"workflow_active: {workflow_active}",
            f"motion_active: {motion_active}",
            f"cmd_vel_subscribers: {cmd_vel_subscribers}",
            "",
            "REFERENCES:",
            f"Slot 1: {'OK' if refs_saved[1] else '-'}",
            f"Slot 2: {'OK' if refs_saved[2] else '-'}",
            f"Slot 3: {'OK' if refs_saved[3] else '-'}",
            f"Slot 4: {'OK' if refs_saved[4] else '-'}",
            f"Slot 5: {'OK' if refs_saved[5] else '-'}",
            "",
            "CHECKS:",
            f"Slot 1: {checks_text[1]}",
            f"Slot 2: {checks_text[2]}",
            f"Slot 3: {checks_text[3]}",
            f"Slot 4: {checks_text[4]}",
            f"Slot 5: {checks_text[5]}",
            "",
            f"Result: {result_text}",
            f"Report: {report_path}",
        ]

        put_lines(left, lines, 10, 28, 27, 0.68, (255, 255, 255))

        if mask is None:
            right = np.zeros_like(left)
        else:
            right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        put_lines(right, ["CARD MASK DEBUG"], 10, 30, 30, 0.8, (255, 255, 255))

        left, right = resize_same_height(left, right)
        combined = np.hstack([left, right])

        return combined

    # ========================================================
    # SLOT LOGIC
    # ========================================================

    def save_slot(self, slot_id):
        with self.lock:
            crop_prepared = self.latest_crop_prepared.copy() if self.latest_crop_prepared is not None else None
            crop_raw = self.latest_crop_raw.copy() if self.latest_crop_raw is not None else None

        if crop_prepared is None or crop_raw is None:
            with self.lock:
                self.status_text = f"SLOT_{slot_id}_NO_CARD"
            self.add_event(f"Save Slot {slot_id}: FAILED | NO_CARD")
            return False, f"Save Slot {slot_id}: NO_CARD"

        ensure_dirs()

        img_path = os.path.join(
            LOG_DIR,
            f"slot_{slot_id}_reference_{now_file_str()}.jpg"
        )
        cv2.imwrite(img_path, crop_raw)

        with self.lock:
            self.references[slot_id] = crop_prepared.copy()
            self.reference_times[slot_id] = now_str()
            self.reference_images[slot_id] = img_path

            self.checks[slot_id] = None
            self.check_times[slot_id] = None

            self.status_text = f"SLOT_{slot_id}_REFERENCE_SAVED"

        self.add_event(f"Save Slot {slot_id}: REFERENCE_SAVED | image={img_path}")
        self.update_result_text()

        return True, f"Save Slot {slot_id}: saved"

    def check_slot(self, slot_id):
        with self.lock:
            ref = self.references[slot_id].copy() if self.references[slot_id] is not None else None
            cur = self.latest_crop_prepared.copy() if self.latest_crop_prepared is not None else None
            crop_raw = self.latest_crop_raw.copy() if self.latest_crop_raw is not None else None

        if ref is None:
            with self.lock:
                self.status_text = f"SLOT_{slot_id}_NO_REFERENCE"
            self.add_event(f"Check Slot {slot_id}: FAILED | NO_REFERENCE")
            return False, f"Check Slot {slot_id}: NO_REFERENCE"

        if cur is None or crop_raw is None:
            with self.lock:
                self.status_text = f"SLOT_{slot_id}_NO_CARD"
            self.add_event(f"Check Slot {slot_id}: FAILED | NO_CARD")
            return False, f"Check Slot {slot_id}: NO_CARD"

        ref_rot = cv2.rotate(ref, cv2.ROTATE_180)

        score_normal = compare_orientation_score(ref, cur)
        score_rot180 = compare_orientation_score(ref_rot, cur)

        verdict = self._orientation_verdict_from_scores(score_normal, score_rot180)

        ensure_dirs()

        img_path = os.path.join(
            LOG_DIR,
            f"slot_{slot_id}_check_{now_file_str()}_{verdict}.jpg"
        )
        cv2.imwrite(img_path, crop_raw)

        # Save side-by-side debug image so user can validate visually
        debug_path = self._save_check_debug_image(
            slot_id, ref, cur, score_normal, score_rot180, verdict
        )

        with self.lock:
            self.checks[slot_id] = {
                "status": verdict,
                "score_normal": score_normal,
                "score_rot180": score_rot180,
                "time": now_str(),
                "image": img_path,
                "debug_image": debug_path,
                "prepared": cur.copy(),  # keep for cross-match analysis at finalize
            }
            self.check_times[slot_id] = now_str()
            self.status_text = f"SLOT_{slot_id}_{verdict}"

        self.add_event(
            f"Check Slot {slot_id}: {verdict} | "
            f"score_normal={score_normal:.3f} | "
            f"score_rot180={score_rot180:.3f} | "
            f"image={img_path} | debug={debug_path}"
        )

        # Re-run cross-match incrementally so the slot panel can remap as
        # checks come in. Cross-match needs at least all 5 references.
        with self.lock:
            all_refs_ready = all(
                self.references[i] is not None
                for i in range(1, NUM_SLOTS + 1)
            )
        if all_refs_ready:
            self.cross_match_result = self.cross_match_analysis()

        self.update_result_text()

        return True, f"Check Slot {slot_id}: {verdict}"

    def get_slot_panel_info(self):
        # Build per-slot view for the UI: each slot shows its reference image
        # and (if available) the matching check image, with rotation status.
        # Cross-match remapping: if cross_match assigned check_slot=X to ref N,
        # then under Slot N we show the image of check_slot=X.
        with self.lock:
            ref_paths = dict(self.reference_images)
            checks_copy = {
                i: dict(self.checks[i]) if self.checks[i] is not None else None
                for i in range(1, NUM_SLOTS + 1)
            }
            cm = self.cross_match_result

        # Build reverse map: ref_slot -> (check_slot, orientation, score)
        reverse_map = {}
        if cm is not None:
            for cs, info in cm.get("per_check", {}).items():
                if info.get("status") != "MATCHED":
                    continue
                rs = info["best_ref_slot"]
                # In case multiple checks claim the same ref_slot, keep the
                # one with highest score
                prev = reverse_map.get(rs)
                if prev is None or info["best_score"] > prev["best_score"]:
                    reverse_map[rs] = info

        slots = []
        rotated_set = set(cm["rotated_refs"]) if cm else set()

        for slot_id in range(1, NUM_SLOTS + 1):
            ref_image = ref_paths.get(slot_id)
            check_image = None
            check_orientation = None
            best_score = None
            origin_check_slot = None

            cm_info = reverse_map.get(slot_id)
            if cm_info is not None:
                origin_check_slot = None
                # find which check_slot mapped here
                for cs, info in cm["per_check"].items():
                    if info.get("status") == "MATCHED" and info["best_ref_slot"] == slot_id:
                        if info["best_score"] == cm_info["best_score"]:
                            origin_check_slot = cs
                            break
                if origin_check_slot is not None:
                    cdata = checks_copy.get(origin_check_slot)
                    if cdata is not None:
                        check_image = cdata.get("image")
                check_orientation = cm_info["best_orientation"]
                best_score = cm_info["best_score"]
            else:
                # No cross-match yet: fall back to direct same-slot check
                cdata = checks_copy.get(slot_id)
                if cdata is not None:
                    check_image = cdata.get("image")
                    if cdata.get("status") == "ROTATED_180":
                        check_orientation = "rot180"
                    elif cdata.get("status") == "NORMAL":
                        check_orientation = "normal"

            slots.append({
                "slot_id": slot_id,
                "reference_image": ref_image,
                "check_image": check_image,
                "check_orientation": check_orientation,
                "best_score": best_score,
                "is_rotated": slot_id in rotated_set,
                "origin_check_slot": origin_check_slot,
            })

        return slots

    def _save_check_debug_image(self, slot_id, ref, check, score_normal, score_rot180, verdict):
        # Side-by-side debug image: [REF normal] [REF rotated 180] [CHECK]
        # so the user can visually confirm what the algorithm is comparing.
        try:
            ref_rot = cv2.rotate(ref, cv2.ROTATE_180)
            h, w = ref.shape
            pad = 8
            label_h = 32
            score_h = 32

            canvas_w = w * 3 + pad * 4
            canvas_h = h + label_h + score_h + pad * 2
            canvas = np.full((canvas_h, canvas_w), 30, dtype=np.uint8)

            y0 = label_h
            canvas[y0:y0 + h, pad:pad + w] = ref
            canvas[y0:y0 + h, pad * 2 + w:pad * 2 + 2 * w] = ref_rot
            canvas[y0:y0 + h, pad * 3 + 2 * w:pad * 3 + 3 * w] = check

            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(canvas_bgr, f"REF Slot {slot_id}", (pad + 5, 24),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas_bgr, "REF rotated 180", (pad * 2 + w + 5, 24),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas_bgr, "CHECK current", (pad * 3 + 2 * w + 5, 24),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            sy = y0 + h + 24
            cv2.putText(canvas_bgr, f"normal={score_normal:.3f}",
                        (pad + 5, sy), font, 0.55, (160, 255, 160), 1, cv2.LINE_AA)
            cv2.putText(canvas_bgr, f"rot180={score_rot180:.3f}",
                        (pad * 2 + w + 5, sy), font, 0.55, (160, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas_bgr, f"-> {verdict}",
                        (pad * 3 + 2 * w + 5, sy), font, 0.55, (255, 220, 100), 1, cv2.LINE_AA)

            path = os.path.join(
                LOG_DIR, f"slot_{slot_id}_check_debug_{now_file_str()}.jpg"
            )
            cv2.imwrite(path, canvas_bgr)
            return path
        except Exception as e:
            self.trace_exc("DEBUG_IMG_ERR", f"slot_{slot_id}")
            return "-"

    def clear_all(self):
        with self.lock:
            self.references = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.reference_times = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.reference_images = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.checks = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.check_times = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.event_log = []
            self.scan_phase = "SAVE_PHASE"  # cancels active wait thread (it polls phase)
            self.ui_phase = "idle"
            self.two_step_active = False
            self.two_step_last_view = "-"
            self.one_shot_active = False
            self.one_shot_last_view = "-"
            self.wait_started_at = None
            self.cross_match_result = None
            self.slot_save_positions = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.canonical_yaw = None
            self.status_text = self._phase_status_text()
            self.result_text = "No rotated card detected yet."
            self.last_report_path = "-"

        ensure_dirs()

        with open(LIVE_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")

        self.add_event("Clear All pressed")

        return True, "Clear All done"


    # ========================================================
    # ONE-SHOT FULL-ROW SCAN MODE (best path for automation)
    # ========================================================

    def _one_shot_reset_data(self):
        # Reset only scan data, not camera/base state. Used when starting a new
        # one-shot run from the button.
        with self.lock:
            self.references = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.reference_times = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.reference_images = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.checks = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.check_times = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.cross_match_result = None
            self.result_text = "No rotated card detected yet."
            self.last_report_path = "-"
            self.one_shot_last_view = "-"

    def _one_shot_pick_five_boxes(self, frame, boxes):
        # Pick the five most plausible cards from a wide frame.  We combine the
        # existing visual quality score with a row-consistency score: the real
        # five cards should sit on roughly the same horizontal row, while walls
        # and edge reflections often sit much higher/lower or at the border.
        boxes = list(boxes)
        if not boxes:
            return []
        if len(boxes) <= ONE_SHOT_REQUIRED_CARDS:
            return sorted(boxes, key=lambda b: b[0] + b[2] / 2.0)

        centers_y = np.array([b[1] + b[3] / 2.0 for b in boxes], dtype=np.float32)
        median_y = float(np.median(centers_y))
        frame_h = float(max(frame.shape[0], 1))
        scored = []
        for b in boxes:
            q = self._two_step_card_quality(frame, b)
            cy = b[1] + b[3] / 2.0
            row_score = 1.0 - min(1.0, abs(cy - median_y) / (0.22 * frame_h))
            score = (
                ONE_SHOT_QUALITY_WEIGHT * float(q.get("quality", 0.0))
                + ONE_SHOT_ROW_Y_WEIGHT * row_score
            )
            scored.append((score, b, q, row_score))

        scored.sort(key=lambda t: t[0], reverse=True)
        picked = [b for _, b, _, _ in scored[:ONE_SHOT_REQUIRED_CARDS]]
        picked.sort(key=lambda b: b[0] + b[2] / 2.0)
        self.trace(
            "1SHOT_PICK",
            "picked=" + str([
                {
                    "bbox": b,
                    "score": round(sc, 3),
                    "q": round(float(q.get("quality", 0.0)), 3),
                    "row": round(rs, 3),
                }
                for sc, b, q, rs in scored[:ONE_SHOT_REQUIRED_CARDS]
            ])
        )
        return picked

    def _draw_one_shot_overview(self, frame, all_boxes, picked_boxes, mode, stamp):
        ensure_dirs()
        overview = frame.copy()
        for idx, bbox in enumerate(all_boxes):
            x, y, bw, bh = bbox
            qd = self._two_step_card_quality(frame, bbox)
            cv2.rectangle(overview, (x, y), (x + bw, y + bh), (0, 210, 255), 2)
            cv2.putText(
                overview, f"det {idx+1} q={qd['quality']:.2f}",
                (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 210, 255), 2, cv2.LINE_AA
            )
        for slot_id, bbox in enumerate(picked_boxes, start=1):
            x, y, bw, bh = bbox
            cv2.rectangle(overview, (x, y), (x + bw, y + bh), (0, 255, 0), 4)
            cv2.putText(
                overview, f"Slot {slot_id}", (x, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA
            )
        overview_path = os.path.join(LOG_DIR, f"one_shot_{mode}_{stamp}_overview.jpg")
        cv2.imwrite(overview_path, overview)
        return overview_path, overview

    def capture_one_shot_view(self, mode):
        # mode: "baseline" or "check". Captures all five cards in one frame.
        # Baseline uses left-to-right slot assignment. Check uses flexible
        # matching, not raw position, so minor robot pose error is acceptable.
        self.wait_for_fresh_frame(max_wait_s=1.2, count=2, min_settle_s=0.20)

        with self.lock:
            frame = self.latest_raw.copy() if self.latest_raw is not None else None

        if frame is None:
            with self.lock:
                self.status_text = f"ONE_SHOT_{mode.upper()}_NO_CAMERA"
            self.add_event(f"One-shot {mode}: FAILED | NO_CAMERA")
            return False, "No camera frame available"

        boxes, mask = self.detect_all_cards(
            frame,
            max_cards=ONE_SHOT_MAX_CANDIDATES,
            edge_reject_frac=ONE_SHOT_EDGE_REJECT_FRAC,
            min_quality=ONE_SHOT_MIN_CARD_QUALITY,
            hard_mask_edge_reject=False,
            allow_edge_touch=True,
        )
        picked_boxes = self._one_shot_pick_five_boxes(frame, boxes)

        stamp = now_file_str()
        overview_path, overview = self._draw_one_shot_overview(frame, boxes, picked_boxes, mode, stamp)
        if mask is not None:
            mask_path = os.path.join(LOG_DIR, f"one_shot_{mode}_{stamp}_mask.jpg")
            cv2.imwrite(mask_path, mask)

        if len(picked_boxes) < ONE_SHOT_REQUIRED_CARDS:
            with self.lock:
                self.status_text = f"ONE_SHOT_{mode.upper()}_NEEDS_5_CARDS | detected={len(boxes)} picked={len(picked_boxes)}"
                self.one_shot_last_view = f"{mode}: detected={len(boxes)}, picked={len(picked_boxes)}, overview={overview_path}"
            self.add_event(
                f"One-shot {mode}: FAILED | need {ONE_SHOT_REQUIRED_CARDS} cards, "
                f"detected={len(boxes)}, picked={len(picked_boxes)} | overview={overview_path}"
            )
            return False, f"One-shot {mode}: need 5 visible cards, detected {len(boxes)}"

        ok_count = 0
        assigned_slots = []
        source_label = f"one_shot_{mode}"

        if mode == "baseline":
            # Baseline defines the identity order: left-to-right = Slot 1..5.
            for slot_id, bbox in enumerate(picked_boxes, start=1):
                x, y, bw, bh = bbox
                crop_raw = frame[y:y + bh, x:x + bw].copy()
                crop_prepared = prepare_crop(crop_raw)
                ok = self._save_slot_from_crop(slot_id, crop_prepared, crop_raw, source_label)
                if ok:
                    ok_count += 1
                    assigned_slots.append(slot_id)
        else:
            # Check is robust: compare each visible crop with every reference.
            for det_idx, bbox in enumerate(picked_boxes, start=1):
                x, y, bw, bh = bbox
                crop_raw = frame[y:y + bh, x:x + bw].copy()
                crop_prepared = prepare_crop(crop_raw)
                ok, matched_slot, best = self._check_crop_flexible_match(
                    crop_prepared, crop_raw, source_label,
                    detection_index=det_idx, frame=frame, bbox=bbox
                )
                if ok and matched_slot is not None:
                    ok_count += 1
                    assigned_slots.append(matched_slot)
                    cv2.putText(
                        overview, f"match Slot {matched_slot}", (x, y + bh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA
                    )
            cv2.imwrite(overview_path, overview)

        with self.lock:
            self.one_shot_last_view = f"{mode}: detected={len(boxes)}, picked={len(picked_boxes)}, assigned={assigned_slots}, ok={ok_count}, overview={overview_path}"
            self.status_text = f"ONE_SHOT_{mode.upper()}_DONE | cards={ok_count}/5"

        self.add_event(
            f"One-shot {mode}: detected={len(boxes)} | picked={len(picked_boxes)} | "
            f"assigned_slots={assigned_slots} | ok_count={ok_count} | overview={overview_path}"
        )
        self.update_result_text()
        return ok_count >= ONE_SHOT_REQUIRED_CARDS, f"One-shot {mode}: {ok_count}/5 cards captured"

    def one_shot_scan_current_view(self):
        # One-button state machine:
        # click 1: baseline all 5 cards, wait 15s, click 2: check all 5 cards + report.
        with self.lock:
            phase = self.scan_phase

        if phase not in ("ONE_BASELINE", "ONE_WAIT", "ONE_CHECK"):
            self._one_shot_reset_data()
            with self.lock:
                self.one_shot_active = True
                self.two_step_active = False
                self.scan_phase = "ONE_BASELINE"
                self.status_text = "ONE_SHOT_BASELINE - show all 5 cards in one clean frame"
            self.add_event("One-shot scan started. Ready for full baseline frame.")
            phase = "ONE_BASELINE"

        if phase == "ONE_BASELINE":
            ok, msg = self.capture_one_shot_view("baseline")
            if not ok:
                return False, msg
            with self.lock:
                missing_refs = [i for i in range(1, NUM_SLOTS + 1) if self.references[i] is None]
            if missing_refs:
                with self.lock:
                    self.status_text = f"ONE_SHOT_BASELINE_INCOMPLETE | missing refs: {missing_refs}"
                return False, f"Baseline incomplete. Missing reference slots: {missing_refs}."
            self._start_one_shot_wait_phase()
            return True, msg + ". All references captured. Flip one card during the wait."

        if phase == "ONE_WAIT":
            return False, "One-shot wait phase active. Flip one card and wait until check phase starts."

        if phase == "ONE_CHECK":
            ok, msg = self.capture_one_shot_view("check")
            if not ok:
                return False, msg
            with self.lock:
                missing_checks = [i for i in range(1, NUM_SLOTS + 1) if self.checks[i] is None]
            if missing_checks:
                with self.lock:
                    self.status_text = f"ONE_SHOT_CHECK_INCOMPLETE | missing checks: {missing_checks}"
                return False, f"Check incomplete. Missing check slots: {missing_checks}. Reposition and scan again."
            self._finalize_session()
            with self.lock:
                self.one_shot_active = False
            return True, msg + f". Done. Result: {self.result_text}"

        return False, f"Unknown one-shot phase: {phase}"

    def _start_one_shot_wait_phase(self):
        with self.lock:
            self.scan_phase = "ONE_WAIT"
            self.wait_started_at = time.time()
            self.status_text = f"ONE_SHOT_WAIT - flip a card now ({int(DEMO_WAIT_SECONDS)}s)"

        self.add_event(f"One-shot wait phase started ({DEMO_WAIT_SECONDS:.0f}s)")
        self.trace("PHASE", f"-> ONE_WAIT for {DEMO_WAIT_SECONDS:.0f}s")
        self.wait_thread = threading.Thread(
            target=self._one_shot_wait_phase_worker,
            name="OneShotWaitPhase",
            daemon=True
        )
        self.wait_thread.start()

    def _one_shot_wait_phase_worker(self):
        start = time.time()
        while time.time() - start < DEMO_WAIT_SECONDS:
            with self.lock:
                if self.scan_phase != "ONE_WAIT":
                    self.trace("PHASE", "One-shot wait phase cancelled (phase changed)")
                    return
                remaining = max(0, int(DEMO_WAIT_SECONDS - (time.time() - start)))
                self.status_text = f"ONE_SHOT_WAIT - flip a card now ({remaining}s remaining)"
            time.sleep(0.5)

        with self.lock:
            if self.scan_phase != "ONE_WAIT":
                return
            self.scan_phase = "ONE_CHECK"
            self.wait_started_at = None
            self.status_text = "ONE_SHOT_CHECK - show all 5 cards again; flexible matching is ON"

        self.add_event("One-shot wait done. Ready for CHECK full frame.")
        self.trace("PHASE", "-> ONE_CHECK")

    def _one_scan_detect_and_pick(self):
        # Helper for auto calibration: read the latest frame, detect all card
        # candidates and pick the best five using the One-Shot picker.
        self.wait_for_fresh_frame(max_wait_s=1.2, count=2, min_settle_s=0.15)
        with self.lock:
            frame = self.latest_raw.copy() if self.latest_raw is not None else None
        if frame is None:
            return None, [], []
        boxes, _mask = self.detect_all_cards(
            frame,
            max_cards=ONE_SHOT_MAX_CANDIDATES,
            edge_reject_frac=ONE_SHOT_EDGE_REJECT_FRAC,
            min_quality=ONE_SHOT_MIN_CARD_QUALITY,
            hard_mask_edge_reject=False,
            allow_edge_touch=True,
        )
        picked = self._one_shot_pick_five_boxes(frame, boxes)
        return frame, boxes, picked

    def one_scan_frame_quality(self, frame, picked_boxes):
        # Returns a compact quality report for the wide 5-card frame.  This is
        # intentionally tolerant: edge cards may be near the border in a one-shot
        # view, but they should not be clipped, and the group should be centered
        # enough that a second scan is comparable to the baseline.
        if frame is None or len(picked_boxes) < ONE_SHOT_REQUIRED_CARDS:
            return {
                "ok": False,
                "reason": f"need_5_cards_got_{len(picked_boxes)}",
                "count": len(picked_boxes),
            }
        h, w = frame.shape[:2]
        xs = [b[0] for b in picked_boxes]
        ys = [b[1] for b in picked_boxes]
        x2s = [b[0] + b[2] for b in picked_boxes]
        y2s = [b[1] + b[3] for b in picked_boxes]
        left, right = min(xs), max(x2s)
        top, bottom = min(ys), max(y2s)
        group_cx = (left + right) / 2.0
        group_w = max(1.0, right - left)
        center_err_ratio = (group_cx - (w / 2.0)) / float(max(w, 1))
        group_w_ratio = group_w / float(max(w, 1))
        margin_left = left / float(max(w, 1))
        margin_right = (w - right) / float(max(w, 1))
        center_ok = abs(center_err_ratio) <= AUTO_ONESCAN_CENTER_TOL_RATIO
        width_ok = (AUTO_ONESCAN_MIN_GROUP_WIDTH_RATIO <= group_w_ratio <= AUTO_ONESCAN_MAX_GROUP_WIDTH_RATIO)
        margin_ok = (margin_left >= AUTO_ONESCAN_SAFE_MARGIN_RATIO and margin_right >= AUTO_ONESCAN_SAFE_MARGIN_RATIO)
        ok = center_ok and width_ok and margin_ok
        reason = "ok" if ok else ",".join([
            r for r, cond in (
                ("center", not center_ok),
                ("width", not width_ok),
                ("margin", not margin_ok),
            ) if cond
        ])
        return {
            "ok": ok,
            "reason": reason,
            "count": len(picked_boxes),
            "center_err_ratio": center_err_ratio,
            "group_w_ratio": group_w_ratio,
            "margin_left": margin_left,
            "margin_right": margin_right,
            "bbox": (int(left), int(top), int(right-left), int(bottom-top)),
        }

    def _score_one_scan_quality_for_sweep(self, q):
        # Higher is better. Count dominates. For equal count, prefer centered,
        # safe-margin, reasonably wide rows. This lets sweep select the best
        # angle even when no angle sees all five cards yet.
        count = int(q.get("count", 0) or 0)
        center = abs(float(q.get("center_err_ratio", 0.0) or 0.0))
        width = float(q.get("group_w_ratio", 0.0) or 0.0)
        ml = float(q.get("margin_left", 0.0) or 0.0)
        mr = float(q.get("margin_right", 0.0) or 0.0)
        margin_penalty = max(0.0, AUTO_ONESCAN_SAFE_MARGIN_RATIO - ml) + max(0.0, AUTO_ONESCAN_SAFE_MARGIN_RATIO - mr)
        # Target width: broad enough to show detail, not clipped.
        target_w = (AUTO_ONESCAN_MIN_GROUP_WIDTH_RATIO + AUTO_ONESCAN_MAX_GROUP_WIDTH_RATIO) / 2.0
        width_penalty = abs(width - target_w)
        return (count * 100.0) - (center * 20.0) - (margin_penalty * 80.0) - (width_penalty * 10.0)

    def turn_to_yaw_blocking(self, target_yaw, label, tolerance_deg=2.5, max_corrections=3,
                             angular_speed=None, accept_residual_deg=None, fail_hard=True):
        # v8 yaw restore: yaw is only a coarse helper. TurtleBot small-angle
        # corrections can overshoot, so we use a slower correction speed and
        # optionally accept a residual error, then let vision sweep finish the job.
        if angular_speed is None:
            angular_speed = AUTO_ONESCAN_YAW_CORRECTION_SPEED
        if accept_residual_deg is None:
            accept_residual_deg = tolerance_deg * 1.8

        last_msg = "Yaw target not attempted"
        for attempt in range(1, max_corrections + 1):
            with self.lock:
                cur = self.current_yaw
                if self.workflow_stop_requested or self.stop_motion_requested:
                    return False, "Yaw restore stopped"
            if cur is None:
                return False, "No yaw available for yaw restore"

            err = shortest_angle_diff(target_yaw, cur)
            err_deg = math.degrees(err)
            if abs(err_deg) <= tolerance_deg:
                self.add_event(f"Yaw target OK: {label} | error={err_deg:+.2f}deg")
                return True, "Yaw target reached"

            self.add_event(
                f"Yaw target correction {label}: attempt {attempt}/{max_corrections} | "
                f"error={err_deg:+.2f}deg | speed={angular_speed:.2f}rad/s"
            )
            ok, msg = self.turn_by_angle_blocking(err, angular_speed, f"{label}_YAW_CORR_{attempt}")
            last_msg = msg
            if not ok:
                # For v8 return alignment we may continue to vision even if a
                # correction times out/fails softly. Hard failures like base death
                # still bubble up from the motion primitive via status/logs.
                self.add_event(f"Yaw target correction failed: {label} | {msg}")
                if fail_hard:
                    return False, msg
                break

        with self.lock:
            cur = self.current_yaw
        final_err = shortest_angle_diff(target_yaw, cur) if cur is not None else 999.0
        final_deg = math.degrees(final_err)
        ok = abs(final_deg) <= accept_residual_deg
        self.add_event(
            f"Yaw target final: {label} | ok={ok} | final_error={final_deg:+.2f}deg | "
            f"accepted_residual={accept_residual_deg:.1f}deg | fail_hard={fail_hard}"
        )
        if ok:
            return True, f"Yaw target close enough (residual {final_deg:+.2f}deg)"
        if fail_hard:
            return False, f"Yaw restore residual error {final_deg:+.2f}deg"
        return True, f"Yaw not exact, continuing to vision sweep (residual {final_deg:+.2f}deg; last={last_msg})"

    def auto_onescan_sweep_for_cards(self, label, sweep_center_yaw, current_q=None):
        # v7: if fewer than 5 cards are visible, do NOT blindly back up.
        # First search left/right in yaw around the expected facing direction.
        # This is exactly the failure mode from v6: after two 180° turns the
        # robot was angled off by ~14°, saw only two cards, then kept backing up.
        if sweep_center_yaw is None:
            return False, "No sweep center yaw", current_q

        best_q = current_q or {"count": 0}
        best_yaw = sweep_center_yaw
        best_score = self._score_one_scan_quality_for_sweep(best_q)

        self.add_event(
            f"Auto one-scan yaw sweep start: {label} | center={math.degrees(sweep_center_yaw):+.1f}deg"
        )

        for deg in AUTO_ONESCAN_SWEEP_DEGREES:
            with self.lock:
                if self.workflow_stop_requested or self.stop_motion_requested:
                    return False, "Sweep stopped", best_q
            target = normalize_angle(sweep_center_yaw + math.radians(float(deg)))
            ok, msg = self.turn_to_yaw_blocking(
                target,
                f"CAL_{label}_SWEEP_{deg:+.0f}DEG",
                tolerance_deg=7.0,
                max_corrections=2,
                angular_speed=AUTO_ONESCAN_YAW_CORRECTION_SPEED,
                accept_residual_deg=18.0,
                fail_hard=False,
            )
            if not ok:
                self.add_event(f"Auto one-scan yaw sweep turn failed: {label} offset={deg:+.0f}deg | {msg}")
                continue
            frame, boxes, picked = self._one_scan_detect_and_pick()
            q = self.one_scan_frame_quality(frame, picked)
            score = self._score_one_scan_quality_for_sweep(q)
            self.add_event(
                f"Auto sweep {label}: offset={deg:+.0f}deg | detected={len(boxes)} | "
                f"picked={len(picked)} | reason={q.get('reason')} | score={score:.1f}"
            )
            self.trace("1SCAN_SWEEP", f"{label} offset={deg:+.0f} q={q} score={score:.2f}")
            if score > best_score:
                best_score = score
                best_q = q
                best_yaw = target
            if q.get("ok"):
                self.add_event(f"Auto one-scan yaw sweep OK: {label} | offset={deg:+.0f}deg")
                return True, "Five-card frame found by yaw sweep", q

        # Return to the best yaw found so later calibration does not continue
        # from a random sweep endpoint.
        self.turn_to_yaw_blocking(
            best_yaw,
            f"CAL_{label}_SWEEP_BEST",
            tolerance_deg=7.0,
            max_corrections=2,
            angular_speed=AUTO_ONESCAN_YAW_CORRECTION_SPEED,
            accept_residual_deg=18.0,
            fail_hard=False,
        )
        self.add_event(
            f"Auto one-scan yaw sweep best: {label} | yaw={math.degrees(best_yaw):+.1f}deg | "
            f"count={best_q.get('count')} | reason={best_q.get('reason')} | score={best_score:.1f}"
        )
        return False, "Sweep did not find a complete five-card frame", best_q

    def auto_calibrate_one_scan_frame(self, label="CHECK", expected_facing_yaw=None):
        # v7 visual calibration for the full one-scan mode.
        # Priority is intentionally changed from v6:
        #   1) Restore/sweep yaw to find the five-card row.
        #   2) Only if five cards are visible, adjust center/size.
        #   3) Distance movement is hard-limited. No endless backing up.
        if not AUTO_ONESCAN_CALIBRATION_ENABLED:
            return True, "Calibration disabled"

        with self.lock:
            entry_yaw = self.current_yaw
        if expected_facing_yaw is None:
            expected_facing_yaw = entry_yaw

        if expected_facing_yaw is not None:
            self.turn_to_yaw_blocking(
                expected_facing_yaw,
                f"CAL_{label}_RESTORE_ENTRY",
                tolerance_deg=AUTO_ONESCAN_RETURN_YAW_TOL_DEG,
                max_corrections=2,
                angular_speed=AUTO_ONESCAN_YAW_CORRECTION_SPEED,
                accept_residual_deg=AUTO_ONESCAN_RETURN_YAW_MAX_RESIDUAL_DEG,
                fail_hard=False,
            )

        self.add_event(f"Auto one-scan calibration start v8: {label}")
        last_report = None
        total_back_m = 0.0
        total_forward_m = 0.0
        sweep_attempted = False

        for step in range(AUTO_ONESCAN_CALIBRATION_MAX_STEPS + 1):
            with self.lock:
                if self.workflow_stop_requested or self.stop_motion_requested:
                    return False, "Calibration stopped"
                self.status_text = f"AUTO_CALIBRATE_{label}_STEP_{step}"

            frame, boxes, picked = self._one_scan_detect_and_pick()
            q = self.one_scan_frame_quality(frame, picked)
            last_report = q
            self.trace("1SCAN_CAL", f"v7 {label} step={step} detected={len(boxes)} picked={len(picked)} q={q}")
            self.add_event(
                f"Auto calibrate v8 {label}: step {step} | detected={len(boxes)} | "
                f"picked={len(picked)} | reason={q.get('reason')} | "
                f"center={q.get('center_err_ratio', 0):+.3f} | "
                f"width={q.get('group_w_ratio', 0):.3f} | "
                f"margins=({q.get('margin_left', 0):.3f},{q.get('margin_right', 0):.3f}) | "
                f"back_total={total_back_m:.2f}m"
            )
            if q.get("ok"):
                self.add_event(f"Auto one-scan calibration OK v8: {label} in {step} steps")
                return True, "Five-card frame calibrated"

            if step >= AUTO_ONESCAN_CALIBRATION_MAX_STEPS:
                break

            count = int(q.get("count", 0) or 0)
            center_err = float(q.get("center_err_ratio", 0.0) or 0.0)
            width_ratio = float(q.get("group_w_ratio", 0.0) or 0.0)
            ml = float(q.get("margin_left", 0.0) or 0.0)
            mr = float(q.get("margin_right", 0.0) or 0.0)

            # Critical v7 fix: fewer than five cards usually means wrong yaw,
            # not wrong distance. Sweep first. Do not back up blindly.
            if count < ONE_SHOT_REQUIRED_CARDS:
                if not sweep_attempted:
                    sweep_attempted = True
                    ok, msg, best_q = self.auto_onescan_sweep_for_cards(label, expected_facing_yaw or entry_yaw, current_q=q)
                    last_report = best_q
                    if ok:
                        return True, msg
                    # Continue one loop from the best yaw found by sweep.
                    continue

                # After a sweep, allow a small back-up ONLY when the partial row
                # clearly appears clipped/wide/at the edge. Otherwise fail safe.
                looks_clipped_or_too_close = (
                    width_ratio > 0.72
                    or ml < (AUTO_ONESCAN_SAFE_MARGIN_RATIO * 0.7)
                    or mr < (AUTO_ONESCAN_SAFE_MARGIN_RATIO * 0.7)
                )
                if looks_clipped_or_too_close and total_back_m + AUTO_ONESCAN_BACKWARD_STEP_M <= AUTO_ONESCAN_MAX_BACK_TOTAL_M:
                    ok, msg = self.move_distance_blocking(
                        -AUTO_ONESCAN_BACKWARD_STEP_M,
                        AUTO_ONESCAN_MOVE_SPEED,
                        f"CAL_{label}_LIMITED_BACK_AFTER_SWEEP_{step+1}"
                    )
                    if not ok:
                        return False, msg
                    total_back_m += AUTO_ONESCAN_BACKWARD_STEP_M
                    # After any distance change, permit another sweep from the
                    # expected yaw because geometry changed.
                    sweep_attempted = False
                    continue

                self.add_event(
                    f"Auto calibrate v8 {label}: FAIL SAFE | only {count}/5 cards after yaw sweep; "
                    f"not backing up blindly. last={q}"
                )
                return False, f"Could not see all 5 cards after yaw sweep: {q}"

            # From here we have all five cards, but the frame may not be centered
            # or sized well. Small turn/forward/back corrections are OK.
            if abs(center_err) > AUTO_ONESCAN_CENTER_TOL_RATIO:
                error_ratio = min(1.0, abs(center_err) / 0.5)
                duration = AUTO_ONESCAN_TURN_DURATION_MIN + (
                    AUTO_ONESCAN_TURN_DURATION_MAX - AUTO_ONESCAN_TURN_DURATION_MIN
                ) * error_ratio
                angular_z = -AUTO_ONESCAN_TURN_SPEED if center_err > 0 else +AUTO_ONESCAN_TURN_SPEED
                ok, msg = self.run_motion_blocking(
                    0.0, angular_z, duration,
                    f"CAL_{label}_CENTER_STEP_{step+1}"
                )
                if not ok:
                    return False, msg
                continue

            if (width_ratio > AUTO_ONESCAN_MAX_GROUP_WIDTH_RATIO
                    or ml < AUTO_ONESCAN_SAFE_MARGIN_RATIO
                    or mr < AUTO_ONESCAN_SAFE_MARGIN_RATIO):
                if total_back_m + AUTO_ONESCAN_BACKWARD_STEP_M > AUTO_ONESCAN_MAX_BACK_TOTAL_M:
                    return False, f"Back-up safety limit reached during calibration: {last_report}"
                ok, msg = self.move_distance_blocking(
                    -AUTO_ONESCAN_BACKWARD_STEP_M,
                    AUTO_ONESCAN_MOVE_SPEED,
                    f"CAL_{label}_BACK_WIDTH_STEP_{step+1}"
                )
                if not ok:
                    return False, msg
                total_back_m += AUTO_ONESCAN_BACKWARD_STEP_M
                continue

            if 0 < width_ratio < AUTO_ONESCAN_MIN_GROUP_WIDTH_RATIO:
                if total_forward_m + AUTO_ONESCAN_FORWARD_STEP_M > AUTO_ONESCAN_MAX_FORWARD_TOTAL_M:
                    # If all 5 cards are visible but a little small, accept rather
                    # than crash or drive too far forward.
                    self.add_event(f"Auto calibrate v8 {label}: forward limit reached; accepting small row")
                    return True, "Five-card frame accepted at forward limit"
                ok, msg = self.move_distance_blocking(
                    +AUTO_ONESCAN_FORWARD_STEP_M,
                    AUTO_ONESCAN_MOVE_SPEED,
                    f"CAL_{label}_FWD_SIZE_STEP_{step+1}"
                )
                if not ok:
                    return False, msg
                total_forward_m += AUTO_ONESCAN_FORWARD_STEP_M
                continue

            # If we get here, quality checker is unhappy for a reason not covered.
            # Fail safe rather than wandering.
            self.add_event(f"Auto calibrate v8 {label}: unhandled quality issue; failing safe. q={q}")
            return False, f"Unhandled calibration quality issue: {q}"

        self.add_event(f"Auto one-scan calibration FAILED v8: {label} | last={last_report}")
        return False, f"Could not calibrate five-card frame: {last_report}"

    def run_full_auto_one_scan_workflow(self):
        # New v8 main flow, bound to the single Start button:
        #   1) Calibrate/capture baseline wide frame (5 cards)
        #   2) Turn 180° away
        #   3) Wait 15 seconds for human to rotate a card
        #   4) Turn 180° back toward cards
        #   5) Auto-calibrate until all 5 cards are visible again
        #   6) Capture check wide frame and report changed card
        self._one_shot_reset_data()
        with self.lock:
            self.scan_phase = "AUTO_ONE_BASELINE"
            self.ui_phase = "calibrate"
            self.one_shot_active = True
            self.two_step_active = False
            self.cross_match_result = None
            self.status_text = "AUTO_ONE_SCAN_STARTING"

        self.add_event("AUTO_ONE_SCAN v8 phase 1/6: calibrating baseline five-card frame")
        ok, msg = self.auto_calibrate_one_scan_frame("BASELINE")
        if not ok:
            return False, msg

        self.add_event("AUTO_ONE_SCAN v8 phase 2/6: capturing baseline")
        self.set_ui_phase("baseline")
        ok, msg = self.capture_one_shot_view("baseline")
        if not ok:
            return False, msg

        with self.lock:
            baseline_facing_yaw = self.current_yaw
        if baseline_facing_yaw is None:
            return False, "No yaw available after baseline capture"
        away_yaw = normalize_angle(baseline_facing_yaw + math.pi)
        self.add_event(
            f"AUTO_ONE_SCAN v8: baseline facing yaw={math.degrees(baseline_facing_yaw):+.2f}deg | "
            f"away target={math.degrees(away_yaw):+.2f}deg"
        )

        self.add_event("AUTO_ONE_SCAN v8 phase 3/6: relaxed 180-degree look-away turn")
        self.set_ui_phase("away")
        # v8 fix: looking away does NOT need precise yaw. v7 failed here by
        # oscillating around the exact 180° target. One odom 180° turn is enough
        # for the human-card-flip phase; the return/check phase is vision-driven.
        ok, msg = self.turn_by_angle_blocking(+math.pi, SLOT_TURN_SPEED, "AUTO_ONE_LOOK_AWAY_RELAXED_180")
        if not ok:
            return False, msg
        with self.lock:
            cur_away = self.current_yaw
        if cur_away is not None:
            away_err = math.degrees(shortest_angle_diff(away_yaw, cur_away))
            self.add_event(
                f"AUTO_ONE_SCAN v8: look-away residual={away_err:+.2f}deg "
                f"(accepted up to {AUTO_ONESCAN_AWAY_ACCEPT_RESIDUAL_DEG:.0f}deg; continuing)"
            )

        with self.lock:
            self.scan_phase = "AUTO_ONE_WAIT"
            self.ui_phase = "wait"
            self.wait_started_at = time.time()
        self.add_event(f"AUTO_ONE_SCAN v8 phase 4/6: waiting {DEMO_WAIT_SECONDS:.0f}s for card rotation")
        ok, msg = self.wait_with_stop_check(DEMO_WAIT_SECONDS, "AUTO_ONE_CARD_ROTATION")
        with self.lock:
            self.wait_started_at = None
        if not ok:
            return False, msg

        self.add_event("AUTO_ONE_SCAN v8 phase 5/6: coarse return to original baseline yaw")
        self.set_ui_phase("return")
        ok, msg = self.turn_to_yaw_blocking(
            baseline_facing_yaw,
            "AUTO_ONE_LOOK_BACK_TO_BASELINE_YAW_COARSE",
            tolerance_deg=AUTO_ONESCAN_RETURN_YAW_TOL_DEG,
            max_corrections=2,
            angular_speed=SLOT_TURN_SPEED,
            accept_residual_deg=AUTO_ONESCAN_RETURN_YAW_MAX_RESIDUAL_DEG,
            fail_hard=False,
        )
        self.add_event(f"AUTO_ONE_SCAN v8: return yaw result | ok={ok} | {msg} | continuing to vision calibration")

        with self.lock:
            self.scan_phase = "AUTO_ONE_CHECK_CALIBRATE"
            self.ui_phase = "check"
        self.add_event("AUTO_ONE_SCAN v8 phase 6/6: yaw-sweep calibrating check frame and scanning")
        ok, msg = self.auto_calibrate_one_scan_frame("CHECK", expected_facing_yaw=baseline_facing_yaw)
        if not ok:
            return False, msg

        with self.lock:
            self.scan_phase = "AUTO_ONE_CHECK"
        ok, msg = self.capture_one_shot_view("check")
        if not ok:
            return False, msg

        with self.lock:
            missing_checks = [i for i in range(1, NUM_SLOTS + 1) if self.checks[i] is None]
        if missing_checks:
            return False, f"Check incomplete. Missing slots: {missing_checks}"

        self._finalize_session()
        with self.lock:
            self.one_shot_active = False
            self.scan_phase = "DONE_PHASE"
            self.ui_phase = "done"
        return True, f"Auto one-scan complete. Result: {self.result_text}"

    # ========================================================
    # TWO-STEP WIDE SCAN MODE (movement-minimal)
    # ========================================================

    def _two_step_reset_data(self):
        # Reset only scan data, not camera/base state. Used when starting a new
        # two-step run from the button.
        with self.lock:
            self.references = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.reference_times = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.reference_images = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.checks = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.check_times = {i: None for i in range(1, NUM_SLOTS + 1)}
            self.cross_match_result = None
            self.result_text = "No rotated card detected yet."
            self.last_report_path = "-"

    def _two_step_pick_slots_and_boxes(self, view_name, boxes):
        # View A: take the left part of the row and assign to slots 1,2,3.
        # View B: take the right part of the row and assign to slots 3,4,5.
        # If View B sees only two cards, assume they are 4,5. This lets the
        # overlap card 3 be optional.
        boxes = list(boxes)
        if view_name == "A":
            slots = list(TWO_STEP_VIEW_A_SLOTS)
            picked_boxes = boxes[:len(slots)]
            picked_slots = slots[:len(picked_boxes)]
            return picked_slots, picked_boxes

        # View B
        if len(boxes) >= 3:
            slots = list(TWO_STEP_VIEW_B_SLOTS)
            picked_boxes = boxes[-len(slots):]
            picked_slots = slots[-len(picked_boxes):]
        elif len(boxes) == 2:
            picked_slots = [4, 5]
            picked_boxes = boxes[-2:]
        elif len(boxes) == 1:
            picked_slots = [5]
            picked_boxes = boxes[-1:]
        else:
            picked_slots = []
            picked_boxes = []
        return picked_slots, picked_boxes

    def _save_slot_from_crop(self, slot_id, crop_prepared, crop_raw, source_label):
        if crop_prepared is None or crop_raw is None or crop_raw.size == 0:
            self.add_event(f"Two-step save Slot {slot_id}: FAILED | bad crop | {source_label}")
            return False

        ensure_dirs()
        img_path = os.path.join(
            LOG_DIR,
            f"slot_{slot_id}_reference_{source_label}_{now_file_str()}.jpg"
        )
        cv2.imwrite(img_path, crop_raw)

        with self.lock:
            self.references[slot_id] = crop_prepared.copy()
            self.reference_times[slot_id] = now_str()
            self.reference_images[slot_id] = img_path
            self.checks[slot_id] = None
            self.check_times[slot_id] = None

        self.add_event(f"Two-step save Slot {slot_id}: REFERENCE_SAVED | {source_label} | image={img_path}")
        return True

    def _orientation_verdict_from_scores(self, score_normal, score_rot180):
        if max(score_normal, score_rot180) < MIN_GOOD_SCORE:
            return "LOW_CONFIDENCE"
        if score_rot180 > score_normal + ROTATION_MARGIN:
            return "ROTATED_180"
        if score_normal > score_rot180 + ROTATION_MARGIN:
            return "NORMAL"
        return "UNCLEAR"

    def _check_crop_flexible_match(self, crop_prepared, crop_raw, source_label, detection_index=None, frame=None, bbox=None):
        # Future auto mode cannot guarantee that the robot captures exactly
        # View A then exactly View B. Therefore, in CHECK mode we ignore the
        # nominal view/slot order. Each visible crop is compared against ALL
        # five references in both orientations; the crop is stored under the
        # best matching reference slot. This makes the check phase robust to:
        #   - View A / View B being swapped
        #   - robot stopping a bit too far left/right
        #   - overlap card appearing in both views
        if crop_prepared is None or crop_raw is None or crop_raw.size == 0:
            self.add_event(f"Two-step flexible check: FAILED | bad crop | {source_label}")
            return False, None, None

        with self.lock:
            refs = {
                i: self.references[i].copy() if self.references[i] is not None else None
                for i in range(1, NUM_SLOTS + 1)
            }
            current_checks = {
                i: dict(self.checks[i]) if self.checks[i] is not None else None
                for i in range(1, NUM_SLOTS + 1)
            }

        q = self._two_step_card_quality(frame, bbox) if frame is not None and bbox is not None else {"quality": 0.50}

        best = None
        for ref_slot, ref in refs.items():
            if ref is None:
                continue
            score_normal, score_rot180 = compare_orientation_pair_multiscale(ref, crop_prepared)
            best_orientation = "normal" if score_normal >= score_rot180 else "rot180"
            best_score = max(score_normal, score_rot180)
            combined_score = (
                TWO_STEP_COMBINED_MATCH_WEIGHT * float(best_score)
                + TWO_STEP_COMBINED_QUALITY_WEIGHT * float(q.get("quality", 0.5))
            )
            cand = {
                "slot_id": ref_slot,
                "score_normal": float(score_normal),
                "score_rot180": float(score_rot180),
                "best_orientation": best_orientation,
                "best_score": float(best_score),
                "quality": float(q.get("quality", 0.5)),
                "combined_score": float(combined_score),
                "quality_detail": q,
            }
            if best is None or cand["combined_score"] > best["combined_score"]:
                best = cand

        if best is None or best["best_score"] < TWO_STEP_MIN_FLEX_MATCH_SCORE:
            self.add_event(
                f"Two-step flexible check: ignored weak crop | {source_label} | "
                f"det={detection_index} | best={best}"
            )
            return False, None, best

        slot_id = best["slot_id"]
        old = current_checks.get(slot_id)
        old_score = -1.0
        old_combined = -1.0
        if old is not None:
            old_score = max(float(old.get("score_normal", -1.0)), float(old.get("score_rot180", -1.0)))
            old_combined = float(old.get("combined_score", old_score))

        # If the same card appears in both views, keep the cleaner crop. v3 uses
        # combined(match + visual quality), not raw match alone, so a slightly
        # higher but edge-cut crop will not overwrite a cleaner one.
        if old is not None and old_combined >= (best["combined_score"] - TWO_STEP_QUALITY_REPLACE_MARGIN):
            self.add_event(
                f"Two-step flexible check: kept existing Slot {slot_id} | "
                f"old_combined={old_combined:.3f} >= new_combined={best['combined_score']:.3f} "
                f"(old_match={old_score:.3f}, new_match={best['best_score']:.3f}, "
                f"new_q={best['quality']:.3f}) | {source_label}"
            )
            return True, slot_id, best

        verdict = self._orientation_verdict_from_scores(best["score_normal"], best["score_rot180"])

        ensure_dirs()
        img_path = os.path.join(
            LOG_DIR,
            f"slot_{slot_id}_check_{source_label}_flex_{now_file_str()}_{verdict}.jpg"
        )
        cv2.imwrite(img_path, crop_raw)

        with self.lock:
            ref_for_debug = self.references[slot_id].copy() if self.references[slot_id] is not None else None
        debug_path = self._save_check_debug_image(
            slot_id, ref_for_debug, crop_prepared,
            best["score_normal"], best["score_rot180"], verdict
        ) if ref_for_debug is not None else "-"

        with self.lock:
            self.checks[slot_id] = {
                "status": verdict,
                "score_normal": best["score_normal"],
                "score_rot180": best["score_rot180"],
                "time": now_str(),
                "image": img_path,
                "debug_image": debug_path,
                "prepared": crop_prepared.copy(),
                "flex_source": source_label,
                "flex_detection_index": detection_index,
                "quality": best["quality"],
                "combined_score": best["combined_score"],
                "quality_detail": best.get("quality_detail", {}),
            }
            self.check_times[slot_id] = now_str()

        self.add_event(
            f"Two-step flexible check: det={detection_index} -> Slot {slot_id}: {verdict} | "
            f"best_orientation={best['best_orientation']} | "
            f"score_normal={best['score_normal']:.3f} | "
            f"score_rot180={best['score_rot180']:.3f} | "
            f"quality={best['quality']:.3f} | combined={best['combined_score']:.3f} | "
            f"image={img_path}"
        )
        return True, slot_id, best

    def _check_slot_from_crop(self, slot_id, crop_prepared, crop_raw, source_label):
        with self.lock:
            ref = self.references[slot_id].copy() if self.references[slot_id] is not None else None

        if ref is None:
            self.add_event(f"Two-step check Slot {slot_id}: FAILED | NO_REFERENCE | {source_label}")
            return False
        if crop_prepared is None or crop_raw is None or crop_raw.size == 0:
            self.add_event(f"Two-step check Slot {slot_id}: FAILED | bad crop | {source_label}")
            return False

        ref_rot = cv2.rotate(ref, cv2.ROTATE_180)
        score_normal = compare_orientation_score(ref, crop_prepared)
        score_rot180 = compare_orientation_score(ref_rot, crop_prepared)

        if max(score_normal, score_rot180) < MIN_GOOD_SCORE:
            verdict = "LOW_CONFIDENCE"
        elif score_rot180 > score_normal + ROTATION_MARGIN:
            verdict = "ROTATED_180"
        elif score_normal > score_rot180 + ROTATION_MARGIN:
            verdict = "NORMAL"
        else:
            verdict = "UNCLEAR"

        ensure_dirs()
        img_path = os.path.join(
            LOG_DIR,
            f"slot_{slot_id}_check_{source_label}_{now_file_str()}_{verdict}.jpg"
        )
        cv2.imwrite(img_path, crop_raw)
        debug_path = self._save_check_debug_image(
            slot_id, ref, crop_prepared, score_normal, score_rot180, verdict
        )

        with self.lock:
            self.checks[slot_id] = {
                "status": verdict,
                "score_normal": score_normal,
                "score_rot180": score_rot180,
                "time": now_str(),
                "image": img_path,
                "debug_image": debug_path,
                "prepared": crop_prepared.copy(),
            }
            self.check_times[slot_id] = now_str()

        self.add_event(
            f"Two-step check Slot {slot_id}: {verdict} | {source_label} | "
            f"score_normal={score_normal:.3f} | score_rot180={score_rot180:.3f} | "
            f"image={img_path} | debug={debug_path}"
        )
        return True

    def capture_two_step_view(self, mode, view_name):
        # mode: "baseline" or "check". Captures all visible cards in current
        # frame, assigns them to slots based on view A/B, then saves/checks.
        self.wait_for_fresh_frame(max_wait_s=1.2, count=2, min_settle_s=0.20)

        with self.lock:
            frame = self.latest_raw.copy() if self.latest_raw is not None else None

        if frame is None:
            with self.lock:
                self.status_text = f"TWO_STEP_{mode.upper()}_{view_name}_NO_CAMERA"
            self.add_event(f"Two-step {mode} View {view_name}: FAILED | NO_CAMERA")
            return False, "No camera frame available"

        boxes, mask = self.detect_all_cards(frame)
        slots, picked_boxes = self._two_step_pick_slots_and_boxes(view_name, boxes)

        ensure_dirs()
        stamp = now_file_str()
        overview = frame.copy()
        for idx, bbox in enumerate(boxes):
            x, y, bw, bh = bbox
            cv2.rectangle(overview, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            qd = self._two_step_card_quality(frame, bbox)
            cv2.putText(overview, f"det {idx+1} q={qd['quality']:.2f}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        for slot_id, bbox in zip(slots, picked_boxes):
            x, y, bw, bh = bbox
            cv2.rectangle(overview, (x, y), (x + bw, y + bh), (0, 255, 0), 4)
            cv2.putText(overview, f"Slot {slot_id}", (x, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        overview_path = os.path.join(LOG_DIR, f"two_step_{mode}_view_{view_name}_{stamp}_overview.jpg")
        cv2.imwrite(overview_path, overview)
        if mask is not None:
            mask_path = os.path.join(LOG_DIR, f"two_step_{mode}_view_{view_name}_{stamp}_mask.jpg")
            cv2.imwrite(mask_path, mask)

        if not picked_boxes:
            with self.lock:
                self.status_text = f"TWO_STEP_{mode.upper()}_{view_name}_NO_CARDS"
            self.add_event(f"Two-step {mode} View {view_name}: NO_CARDS | overview={overview_path}")
            return False, "No cards detected in this view"

        if mode == "baseline":
            min_needed = TWO_STEP_MIN_BASELINE_A_CARDS if view_name == "A" else TWO_STEP_MIN_BASELINE_B_CARDS
            if len(picked_boxes) < min_needed:
                self.add_event(
                    f"Two-step baseline View {view_name}: weak view | "
                    f"picked={len(picked_boxes)} < min_needed={min_needed} | overview={overview_path}"
                )
                # Still allow the capture if it helps fill missing refs later,
                # but make the status/report explicit. The final missing-ref
                # gate will block completion if coverage is not enough.

        ok_count = 0
        source_label = f"two_step_{mode}_view_{view_name}"
        assigned_slots = []

        if mode == "check" and TWO_STEP_CHECK_FLEXIBLE_MATCH:
            # CHECK is intentionally view-order independent: use every real
            # detection in this view, match it to the best reference, and keep
            # the best crop for each reference slot. This is the key change for
            # later automatic movement.
            for det_idx, bbox in enumerate(boxes, start=1):
                x, y, bw, bh = bbox
                crop_raw = frame[y:y + bh, x:x + bw].copy()
                crop_prepared = prepare_crop(crop_raw)
                ok, matched_slot, best = self._check_crop_flexible_match(
                    crop_prepared, crop_raw, source_label, detection_index=det_idx,
                    frame=frame, bbox=bbox
                )
                if ok and matched_slot is not None:
                    ok_count += 1
                    assigned_slots.append(matched_slot)
                    cv2.putText(overview, f"match Slot {matched_slot}", (x, y + bh - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)
        else:
            for slot_id, bbox in zip(slots, picked_boxes):
                x, y, bw, bh = bbox
                crop_raw = frame[y:y + bh, x:x + bw].copy()
                crop_prepared = prepare_crop(crop_raw)
                if mode == "baseline":
                    ok = self._save_slot_from_crop(slot_id, crop_prepared, crop_raw, source_label)
                else:
                    ok = self._check_slot_from_crop(slot_id, crop_prepared, crop_raw, source_label)
                if ok:
                    ok_count += 1
                    assigned_slots.append(slot_id)

        # Re-save overview after flexible labels were added.
        cv2.imwrite(overview_path, overview)

        with self.lock:
            self.two_step_last_view = f"{mode} {view_name}: detected={len(boxes)}, assigned={assigned_slots}, saved_checked={ok_count}, overview={overview_path}"
            self.status_text = f"TWO_STEP_{mode.upper()}_{view_name}_DONE | cards={ok_count}"

        self.add_event(
            f"Two-step {mode} View {view_name}: detected={len(boxes)} | "
            f"assigned_slots={assigned_slots if mode == 'check' else slots} | ok_count={ok_count} | overview={overview_path}"
        )
        self.update_result_text()
        return ok_count > 0, f"Two-step {mode} View {view_name}: {ok_count} card(s) captured"

    def two_step_scan_current_view(self):
        # One-button state machine for the new experiment:
        # click 1: baseline View A, click 2: baseline View B,
        # wait 15s, click 3: check View A, click 4: check View B + report.
        with self.lock:
            phase = self.scan_phase

        if phase not in ("TWO_BASELINE_A", "TWO_BASELINE_B", "TWO_WAIT", "TWO_CHECK_A", "TWO_CHECK_B"):
            self._two_step_reset_data()
            with self.lock:
                self.two_step_active = True
                self.one_shot_active = False
                self.scan_phase = "TWO_BASELINE_A"
                self.status_text = "TWO_STEP_BASELINE_A - show a clean first view (ideally cards 1-3)"
            self.add_event("Two-step scan started. Ready for baseline View A.")
            phase = "TWO_BASELINE_A"

        if phase == "TWO_BASELINE_A":
            ok, msg = self.capture_two_step_view("baseline", "A")
            if ok:
                with self.lock:
                    self.scan_phase = "TWO_BASELINE_B"
                    self.status_text = "TWO_STEP_BASELINE_B - show second view to complete all 5 references"
                return True, msg + ". Now capture baseline View B."
            return False, msg

        if phase == "TWO_BASELINE_B":
            ok, msg = self.capture_two_step_view("baseline", "B")
            with self.lock:
                missing_refs = [i for i in range(1, NUM_SLOTS + 1) if self.references[i] is None]
            if not ok:
                return False, msg
            if missing_refs:
                with self.lock:
                    self.status_text = f"TWO_STEP_BASELINE_INCOMPLETE | missing refs: {missing_refs}"
                return False, f"Baseline incomplete. Missing reference slots: {missing_refs}. Reposition and try View B again."
            self._start_two_step_wait_phase()
            return True, msg + ". All references captured. Flip one card during the wait."

        if phase == "TWO_WAIT":
            return False, "Two-step wait phase active. Flip one card and wait until check phase starts."

        if phase == "TWO_CHECK_A":
            ok, msg = self.capture_two_step_view("check", "A")
            if ok:
                with self.lock:
                    self.scan_phase = "TWO_CHECK_B"
                    self.status_text = "TWO_STEP_CHECK_B - show any second view; flexible matching + crop quality is ON"
                return True, msg + ". Now capture check View B."
            return False, msg

        if phase == "TWO_CHECK_B":
            ok, msg = self.capture_two_step_view("check", "B")
            with self.lock:
                missing_checks = [i for i in range(1, NUM_SLOTS + 1) if self.checks[i] is None]
            if not ok:
                return False, msg
            if missing_checks:
                with self.lock:
                    self.status_text = f"TWO_STEP_CHECK_INCOMPLETE | missing checks: {missing_checks}"
                return False, f"Check incomplete. Missing check slots: {missing_checks}. Reposition and try View B again."
            self._finalize_session()
            with self.lock:
                self.two_step_active = False
            return True, msg + f". Done. Result: {self.result_text}"

        return False, f"Unknown two-step phase: {phase}"

    def _start_two_step_wait_phase(self):
        with self.lock:
            self.scan_phase = "TWO_WAIT"
            self.wait_started_at = time.time()
            self.status_text = f"TWO_STEP_WAIT - flip a card now ({int(DEMO_WAIT_SECONDS)}s)"

        self.add_event(f"Two-step wait phase started ({DEMO_WAIT_SECONDS:.0f}s)")
        self.wait_thread = threading.Thread(
            target=self._two_step_wait_phase_worker,
            name="TwoStepWaitPhase",
            daemon=True
        )
        self.wait_thread.start()

    def _two_step_wait_phase_worker(self):
        start = time.time()
        while time.time() - start < DEMO_WAIT_SECONDS:
            with self.lock:
                if self.scan_phase != "TWO_WAIT":
                    self.trace("PHASE", "Two-step wait cancelled (phase changed)")
                    return
                remaining = max(0, int(DEMO_WAIT_SECONDS - (time.time() - start)))
                self.status_text = f"TWO_STEP_WAIT - flip a card now ({remaining}s remaining)"
            time.sleep(0.5)

        with self.lock:
            if self.scan_phase != "TWO_WAIT":
                return
            self.scan_phase = "TWO_CHECK_A"
            self.wait_started_at = None
            self.status_text = "TWO_STEP_CHECK_A - show any useful view; flexible matching + crop quality is ON"
        self.add_event("Two-step wait done. Ready for CHECK View A.")
        self.trace("PHASE", "-> TWO_CHECK_A")

    # ========================================================
    # MANUAL SCAN STATE MACHINE (movement disabled)
    # ========================================================

    def _phase_status_text(self):
        # Build the user-facing status string from current phase + state.
        # Called whenever phase transitions or after each scan click.
        with self.lock:
            phase = self.scan_phase
            missing_refs = [i for i in range(1, NUM_SLOTS + 1) if self.references[i] is None]
            missing_checks = [i for i in range(1, NUM_SLOTS + 1) if self.checks[i] is None]
            wait_at = self.wait_started_at

        if phase == "SAVE_PHASE":
            if missing_refs:
                return f"SAVE_PHASE - Place robot at Slot {missing_refs[0]} and click Scan"
            return "SAVE_PHASE - All references saved (transitioning to wait)"

        if phase == "WAIT_PHASE":
            if wait_at is None:
                return "WAIT_PHASE - flip a card now"
            remaining = max(0, int(DEMO_WAIT_SECONDS - (time.time() - wait_at)))
            return f"WAIT_PHASE - flip a card now ({remaining}s remaining)"

        if phase == "CHECK_PHASE":
            if missing_checks:
                return f"CHECK_PHASE - Place robot at Slot {missing_checks[0]} and click Scan"
            return "CHECK_PHASE - All checks saved (finalizing)"

        if phase == "DONE_PHASE":
            return "DONE - Click Clear to start again"

        return phase

    def scan_current_slot(self):
        # Single click: scan whatever card is currently in view, save as the
        # next missing slot (reference or check, depending on phase).
        with self.lock:
            phase = self.scan_phase
            missing_refs = [i for i in range(1, NUM_SLOTS + 1) if self.references[i] is None]
            missing_checks = [i for i in range(1, NUM_SLOTS + 1) if self.checks[i] is None]

        if phase == "SAVE_PHASE":
            if not missing_refs:
                # already complete; trigger wait
                self._start_wait_phase()
                return True, "All references already saved; wait phase started."

            slot_id = missing_refs[0]
            ok, msg = self.save_slot(slot_id)
            if not ok:
                with self.lock:
                    self.status_text = f"SAVE_FAILED_SLOT_{slot_id}: {msg}"
                return False, msg

            # update phase / status
            with self.lock:
                still_missing = [
                    i for i in range(1, NUM_SLOTS + 1)
                    if self.references[i] is None
                ]

            if not still_missing:
                self._start_wait_phase()
                return True, f"Slot {slot_id} saved. All 5 references done. Wait phase started."

            with self.lock:
                self.status_text = self._phase_status_text()
            return True, f"Slot {slot_id} saved. Move to Slot {still_missing[0]} and click Scan."

        if phase == "WAIT_PHASE":
            return False, "Wait phase active. Wait for the timer to finish before clicking Scan."

        if phase == "CHECK_PHASE":
            if not missing_checks:
                self._finalize_session()
                return True, "All checks already done."

            slot_id = missing_checks[0]
            ok, msg = self.check_slot(slot_id)
            if not ok:
                with self.lock:
                    self.status_text = f"CHECK_FAILED_SLOT_{slot_id}: {msg}"
                return False, msg

            with self.lock:
                still_missing = [
                    i for i in range(1, NUM_SLOTS + 1)
                    if self.checks[i] is None
                ]

            if not still_missing:
                self._finalize_session()
                with self.lock:
                    return True, f"Slot {slot_id} checked. All done. Result: {self.result_text}"

            with self.lock:
                self.status_text = self._phase_status_text()
            return True, f"Slot {slot_id} checked. Move to Slot {still_missing[0]} and click Scan."

        if phase == "DONE_PHASE":
            return False, "Magic trick already complete. Click Clear to restart."

        return False, f"Unknown phase: {phase}"

    def _start_wait_phase(self):
        with self.lock:
            self.scan_phase = "WAIT_PHASE"
            self.wait_started_at = time.time()
            self.status_text = self._phase_status_text()

        self.add_event(f"Wait phase started ({DEMO_WAIT_SECONDS:.0f}s)")
        self.trace("PHASE", f"-> WAIT_PHASE for {DEMO_WAIT_SECONDS:.0f}s")

        self.wait_thread = threading.Thread(
            target=self._wait_phase_worker,
            name="WaitPhase",
            daemon=True
        )
        self.wait_thread.start()

    def _wait_phase_worker(self):
        start = time.time()
        while time.time() - start < DEMO_WAIT_SECONDS:
            with self.lock:
                if self.scan_phase != "WAIT_PHASE":
                    self.trace("PHASE", "Wait phase cancelled (phase changed)")
                    return
                self.status_text = self._phase_status_text()
            time.sleep(0.5)

        with self.lock:
            if self.scan_phase != "WAIT_PHASE":
                return
            self.scan_phase = "CHECK_PHASE"
            self.wait_started_at = None
            self.status_text = self._phase_status_text()

        self.add_event("Wait phase done. Ready for CHECK_PHASE.")
        self.trace("PHASE", "-> CHECK_PHASE")

    def _finalize_session(self):
        # Cross-match has already been running incrementally after each check;
        # do one final pass to be sure, then save the report.
        self.cross_match_result = self.cross_match_analysis()
        self.update_result_text()
        report_path = self.save_report()
        with self.lock:
            self.scan_phase = "DONE_PHASE"
            self.last_report_path = report_path
            self.status_text = self._phase_status_text()
        self.add_event(f"Session complete. Report: {report_path}")
        self.trace("PHASE", f"-> DONE_PHASE (report={report_path})")

    def correct_yaw_to_canonical(self, label, threshold_deg=2.5):
        # Snap robot heading back to self.canonical_yaw if it has drifted
        # by more than threshold_deg. No-op if canonical_yaw not set or
        # current_yaw unavailable. This is what bounds per-transition
        # angular drift so it doesn't accumulate over 4-8 step movements.
        with self.lock:
            target = self.canonical_yaw
            cur = self.current_yaw
        if target is None or cur is None:
            return True, "Yaw correction skipped (no reference)"

        error = shortest_angle_diff(target, cur)
        if abs(error) < math.radians(threshold_deg):
            return True, f"Yaw OK (drift {math.degrees(error):+.2f}° < {threshold_deg}°)"

        self.add_event(
            f"Yaw correction {label}: drift {math.degrees(error):+.2f}° -> snapping back"
        )
        return self.turn_by_angle_blocking(
            error, SLOT_TURN_SPEED, f"{label}_YAW_SNAP"
        )

    def _auto_run_save_phase(self):
        # Save references for slots 1..5 with closed-loop motion and vision
        # alignment between slots. Robot must START in front of slot 1.
        with self.lock:
            self.canonical_yaw = None  # will be captured at slot 1

        for slot_id in range(1, NUM_SLOTS + 1):
            with self.lock:
                if self.workflow_stop_requested:
                    return False, "Stopped"

            self.add_event(f"AUTO_RUN save: slot {slot_id}/5")

            # On every slot, do a vision align before save (the FIRST slot
            # too — user might place robot slightly off, vision corrects).
            ok, msg = self.vision_align_to_card(slot_id, "AUTO_SAVE")
            if not ok:
                self.add_event(f"AUTO_RUN save FAILED: align slot {slot_id}: {msg}")
                return False, msg

            # CAPTURE the canonical "facing cards" yaw at slot 1, AFTER
            # vision_align settles. All later transitions will snap heading
            # back to this value to prevent accumulated angular drift.
            if slot_id == 1:
                with self.lock:
                    self.canonical_yaw = self.current_yaw
                if self.canonical_yaw is not None:
                    self.add_event(
                        f"AUTO_RUN: canonical yaw captured = "
                        f"{math.degrees(self.canonical_yaw):+.2f}°"
                    )

            # Final settle before save: 2 fresh frames + small extra wait so
            # the captured JPEG is from a fully-stable scene, even when
            # vision_align took 0 corrective steps.
            self.wait_for_fresh_frame(max_wait_s=1.0, count=2, min_settle_s=0.20)
            ok, msg = self.save_slot(slot_id)
            if not ok:
                self.add_event(f"AUTO_RUN save FAILED: save slot {slot_id}: {msg}")
                return False, msg

            # Record exact odom position at save — used by check phase to
            # compute the ACTUAL slot-to-slot distance (which may differ
            # from SLOT_FORWARD_DISTANCE_M depending on user setup).
            with self.lock:
                px = self.current_pos_x
                py = self.current_pos_y
            if px is not None and py is not None:
                self.slot_save_positions[slot_id] = (px, py)
                self.add_event(
                    f"AUTO_RUN save: slot {slot_id} position recorded at "
                    f"({px:.3f}, {py:.3f})"
                )

            # Step to next slot (no step after slot 5)
            if slot_id < NUM_SLOTS:
                # CRITICAL: snap yaw to canonical BEFORE the step. Vision_align
                # may have rotated the robot to center the card, so its current
                # yaw is canonical + correction. If we step from that, the
                # forward motion would go in a tilted direction (creating
                # lateral drift). By snapping first, the step's forward
                # motion goes in the perpendicular direction.
                self.correct_yaw_to_canonical(
                    f"BEFORE_STEP_NEXT_FROM_SLOT_{slot_id}"
                )

                # Use measured slot gap if we have a previous measurement.
                # After slot 2 we know gap 1->2, use it as estimate for the
                # next gap. Subtract 1cm safety margin and clamp to safe range.
                distance_m = None
                if slot_id >= 2:
                    prev_pos = self.slot_save_positions.get(slot_id - 1)
                    cur_pos = self.slot_save_positions.get(slot_id)
                    if prev_pos is not None and cur_pos is not None:
                        last_gap = math.sqrt(
                            (prev_pos[0] - cur_pos[0]) ** 2
                            + (prev_pos[1] - cur_pos[1]) ** 2
                        )
                        distance_m = max(0.10, min(0.35, last_gap - 0.01))
                        self.add_event(
                            f"AUTO_RUN save: slot {slot_id} -> {slot_id + 1} | "
                            f"using measured gap from prev step "
                            f"({last_gap:.3f}m) -> moving {distance_m:.3f}m"
                        )

                ok, msg = self.step_to_next_slot_workflow(
                    f"AUTO_STEP_NEXT_AFTER_SLOT_{slot_id}",
                    distance_m=distance_m,
                )
                if not ok:
                    return False, msg

                # Snap again AFTER the step to bound per-transition drift.
                self.correct_yaw_to_canonical(
                    f"AFTER_STEP_NEXT_TO_SLOT_{slot_id + 1}"
                )

        return True, "Save phase done"

    def _auto_run_check_phase(self):
        # Check slots 5..1 in reverse. Robot must be in front of slot 5
        # (after the 90° turn-back from look-away).
        # Snap heading to canonical BEFORE the first check — the two 90°
        # turn-aways for the wait phase may have accumulated drift.
        self.correct_yaw_to_canonical("CHECK_PHASE_START")

        for index, slot_id in enumerate(range(NUM_SLOTS, 0, -1)):
            with self.lock:
                if self.workflow_stop_requested:
                    return False, "Stopped"

            self.add_event(f"AUTO_RUN check: slot {slot_id} ({index + 1}/{NUM_SLOTS})")

            ok, msg = self.vision_align_to_card(slot_id, "AUTO_CHECK")
            if not ok:
                self.add_event(f"AUTO_RUN check FAILED: align slot {slot_id}: {msg}")
                return False, msg

            self.wait_for_fresh_frame(max_wait_s=1.0, count=2, min_settle_s=0.20)
            ok, msg = self.check_slot(slot_id)
            if not ok:
                self.add_event(f"AUTO_RUN check FAILED: check slot {slot_id}: {msg}")
                return False, msg

            if slot_id > 1:
                # Snap yaw BEFORE step (same reason as save phase: vision_align
                # may have tilted us, so step needs to start from canonical).
                self.correct_yaw_to_canonical(
                    f"BEFORE_STEP_PREV_FROM_SLOT_{slot_id}"
                )

                # Use the actual measured distance between current slot and
                # previous slot, recorded during save phase. This handles
                # variable user spacing (e.g. 18cm between some slots, 22cm
                # between others). Falls back to default if recording missing.
                cur_pos = self.slot_save_positions.get(slot_id)
                prev_pos = self.slot_save_positions.get(slot_id - 1)
                if cur_pos is not None and prev_pos is not None:
                    measured = math.sqrt(
                        (cur_pos[0] - prev_pos[0]) ** 2
                        + (cur_pos[1] - prev_pos[1]) ** 2
                    )
                    # Subtract a tiny stop-margin tolerance and clamp to
                    # a reasonable range so an outlier reading can't push
                    # the robot unsafely far.
                    distance_m = max(0.10, min(0.35, measured - 0.01))
                    self.add_event(
                        f"AUTO_RUN check: slot {slot_id} -> {slot_id - 1} | "
                        f"using measured gap {measured:.3f}m -> moving "
                        f"{distance_m:.3f}m"
                    )
                else:
                    distance_m = None  # default
                    self.add_event(
                        f"AUTO_RUN check: slot {slot_id} -> {slot_id - 1} | "
                        f"no recorded gap, using default "
                        f"{SLOT_FORWARD_DISTANCE_M}m"
                    )

                ok, msg = self.step_to_previous_slot_workflow(
                    f"AUTO_STEP_PREV_AFTER_SLOT_{slot_id}",
                    distance_m=distance_m
                )
                if not ok:
                    return False, msg

                # Snap heading back to canonical (same as save phase).
                # Bounds the per-transition drift that was causing slot 2/1
                # to capture cards at perspective angles.
                self.correct_yaw_to_canonical(
                    f"AFTER_STEP_PREV_TO_SLOT_{slot_id - 1}"
                )

        return True, "Check phase done"

    def cross_match_analysis(self):
        # For each check, compare against ALL 5 references in BOTH orientations.
        # Returns dict with:
        #   per_check: {check_slot: {best_ref, best_orientation, best_score, all_scores}}
        #   summary:   text describing rotated/moved cards
        #
        # This makes the system robust to two common user mistakes:
        # 1. Reverse-order scanning (check 5->1 instead of 1->5)
        # 2. Mismapping which slot the user meant
        # And it explicitly identifies rotation by direct comparison against
        # rotated reference, even if the per-slot check missed it.
        with self.lock:
            refs = {
                i: self.references[i].copy() if self.references[i] is not None else None
                for i in range(1, NUM_SLOTS + 1)
            }
            checks = {}
            for i in range(1, NUM_SLOTS + 1):
                c = self.checks[i]
                if c is not None and "prepared" in c and c["prepared"] is not None:
                    checks[i] = c["prepared"].copy()
                else:
                    checks[i] = None

        # Step 1: compute all scores (each check vs each ref, both orientations)
        all_scores_per_check = {}
        for check_slot in range(1, NUM_SLOTS + 1):
            check = checks[check_slot]
            if check is None:
                all_scores_per_check[check_slot] = None
                continue

            ref_scores = {}
            for ref_slot in range(1, NUM_SLOTS + 1):
                ref = refs[ref_slot]
                if ref is None:
                    continue
                # Multi-scale orientation comparison: tries several inner
                # crops, picks the margin with the strongest rotation signal.
                s_normal, s_rot = compare_orientation_pair_multiscale(ref, check)
                ref_scores[ref_slot] = {
                    "normal": float(s_normal),
                    "rot180": float(s_rot),
                }
            all_scores_per_check[check_slot] = ref_scores

        # Step 2: greedy 1-to-1 assignment so each ref is claimed by at most
        # one check. Pick the highest-scoring check<->ref pair first, then
        # remove that check and ref from consideration, repeat.
        candidates = []
        for cs, ref_scores in all_scores_per_check.items():
            if ref_scores is None:
                continue
            for rs, scores in ref_scores.items():
                # Best of (normal, rot180) for this pair determines its claim
                if scores["normal"] >= scores["rot180"]:
                    candidates.append({
                        "check_slot": cs, "ref_slot": rs,
                        "orientation": "normal",
                        "score": scores["normal"],
                    })
                else:
                    candidates.append({
                        "check_slot": cs, "ref_slot": rs,
                        "orientation": "rot180",
                        "score": scores["rot180"],
                    })

        candidates.sort(key=lambda c: c["score"], reverse=True)

        assigned_check = set()
        assigned_ref = set()
        assignments = {}  # check_slot -> {ref_slot, orientation, score}
        for c in candidates:
            if c["check_slot"] in assigned_check:
                continue
            if c["ref_slot"] in assigned_ref:
                continue
            assignments[c["check_slot"]] = c
            assigned_check.add(c["check_slot"])
            assigned_ref.add(c["ref_slot"])
            if len(assignments) == NUM_SLOTS:
                break

        # Step 3: build per_check dict from assignments
        per_check = {}
        for check_slot in range(1, NUM_SLOTS + 1):
            ref_scores = all_scores_per_check.get(check_slot)
            if ref_scores is None:
                per_check[check_slot] = {"status": "NO_CHECK"}
                continue

            assign = assignments.get(check_slot)
            if assign is None:
                per_check[check_slot] = {
                    "status": "UNASSIGNED",
                    "all_scores": ref_scores,
                }
                continue

            per_check[check_slot] = {
                "status": "MATCHED",
                "best_ref_slot": assign["ref_slot"],
                "best_orientation": assign["orientation"],
                "best_score": assign["score"],
                "all_scores": ref_scores,
            }

        # Summarize
        # Two independent decisions per check:
        # 1) Best matching reference (any slot, any orientation): tells us about
        #    slot order/swap — uses absolute score with MIN_GOOD_SCORE.
        # 2) Was the card rotated? Look at scores AT the best-ref-slot in both
        #    orientations. Rotation is only flagged when ALL of the following
        #    are true:
        #      - rot180 > normal + ROTATION_MARGIN   (rot180 actually wins)
        #      - rot180 > ROTATION_NOISE_FLOOR        (signal above noise)
        #      - rot180 >= ROTATION_HIGH_CONF_ABS  OR
        #        (rot180 - normal) >= ROTATION_HIGH_CONF_DELTA
        #          (i.e. either the absolute match is strong, or the gap is
        #           clearly meaningful — protects against false-positive
        #           rotation flags on poorly-framed scans where both scores
        #           are low and the rot180 lead is just noise.)
        ROTATION_NOISE_FLOOR = 0.10
        ROTATION_HIGH_CONF_ABS = 0.45
        ROTATION_HIGH_CONF_DELTA = 0.15

        rotated_refs = []   # ref slot ids that were rotated
        moved = []          # (check_slot, ref_slot) pairs where mismatched
        low_conf = []

        for check_slot, info in per_check.items():
            if info.get("status") != "MATCHED":
                continue
            score = info["best_score"]
            ref_slot = info["best_ref_slot"]
            scores_at_best = info["all_scores"][ref_slot]
            normal_s = scores_at_best["normal"]
            rot_s = scores_at_best["rot180"]

            # Rotation: relative comparison at best-matching reference, plus
            # a confidence gate to suppress false positives on bad scans.
            rotation_detected = (
                rot_s > normal_s + ROTATION_MARGIN
                and rot_s > ROTATION_NOISE_FLOOR
                and (
                    rot_s >= ROTATION_HIGH_CONF_ABS
                    or (rot_s - normal_s) >= ROTATION_HIGH_CONF_DELTA
                )
            )

            if rotation_detected:
                rotated_refs.append(ref_slot)
                if ref_slot != check_slot:
                    moved.append((check_slot, ref_slot))
                continue

            # Not rotated: need absolute confidence in the normal match
            if score < MIN_GOOD_SCORE:
                low_conf.append(check_slot)
                continue

            if ref_slot != check_slot:
                moved.append((check_slot, ref_slot))

        lines = []
        if rotated_refs:
            if len(rotated_refs) == 1:
                lines.append(f"Card {rotated_refs[0]} was rotated 180 degrees.")
            else:
                joined = ", ".join(f"Card {r}" for r in rotated_refs)
                lines.append(f"Rotated cards: {joined}.")
        elif not low_conf:
            lines.append("No rotated card detected. All cards match their references.")

        if moved:
            mapping = ", ".join(f"check[{c}]=ref[{r}]" for c, r in moved)
            lines.append(f"(Scan order detail: {mapping})")
        if low_conf and not rotated_refs:
            lines.append(
                f"Low-confidence checks at scan position(s): "
                f"{', '.join(str(s) for s in low_conf)} "
                f"(re-scan or improve framing to confirm)"
            )

        summary = " | ".join(lines)

        self.add_event(f"Cross-match: {summary}")
        self.trace("CROSSMATCH", f"per_check={per_check}")
        self.trace("CROSSMATCH", f"summary={summary}")

        return {
            "per_check": per_check,
            "summary": summary,
            "rotated_refs": rotated_refs,
            "moved": moved,
            "low_conf": low_conf,
        }

    def update_result_text(self):
        # Prefer the cross-match summary when available (more accurate; handles
        # reverse-order scanning and rotation against any reference).
        # Fall back to per-slot verdicts otherwise.
        with self.lock:
            cm = self.cross_match_result

        if cm is not None:
            with self.lock:
                self.result_text = cm["summary"]
            return

        with self.lock:
            rotated_slots = []
            for i in range(1, NUM_SLOTS + 1):
                c = self.checks[i]
                if c is not None and c["status"] == "ROTATED_180":
                    rotated_slots.append(i)

            if len(rotated_slots) == 0:
                self.result_text = "No rotated card detected yet."
            elif len(rotated_slots) == 1:
                self.result_text = f"Changed card detected: Slot {rotated_slots[0]} is ROTATED_180."
            else:
                joined = ", ".join(str(x) for x in rotated_slots)
                self.result_text = f"Multiple rotated cards detected: Slot {joined}."

    # ========================================================
    # DEMO WORKFLOW
    # ========================================================

    def wait_for_stable_card(self, timeout=CARD_WAIT_TIMEOUT):
        start = time.time()
        stable_start = None

        while time.time() - start < timeout:
            with self.lock:
                stop_requested = self.workflow_stop_requested or self.stop_motion_requested
                found = self.latest_card_found

            if stop_requested:
                return False, "STOP_REQUESTED"

            if found:
                if stable_start is None:
                    stable_start = time.time()
                elif time.time() - stable_start >= CARD_STABLE_SECONDS:
                    return True, "CARD_STABLE"
            else:
                stable_start = None

            time.sleep(0.05)

        return False, "CARD_WAIT_TIMEOUT"

    def wait_with_stop_check(self, duration, label):
        self.add_event(f"Workflow wait start: {label} | duration={duration:.1f}s")
        start = time.time()

        while time.time() - start < duration:
            with self.lock:
                if self.workflow_stop_requested or self.stop_motion_requested:
                    self.add_event(f"Workflow wait stopped: {label}")
                    return False, "Wait stopped"
                remaining = duration - (time.time() - start)
                self.status_text = f"WORKFLOW_WAIT_{label}_{max(0, int(remaining))}S"

            time.sleep(0.1)

        self.add_event(f"Workflow wait end: {label}")
        return True, f"Wait done: {label}"

    def run_motion_blocking(self, linear_x, angular_z, duration, label):
        if self.cmd_vel_subscriber_count() <= 0:
            with self.lock:
                self.status_text = "NO_CMD_VEL_SUBSCRIBER"
            self.add_event(f"Workflow motion rejected: {label} | no /cmd_vel subscriber")
            return False, "No /cmd_vel subscriber. Is turtlebot3_bringup running?"

        if not self.base_alive():
            with self.lock:
                self.status_text = "BASE_DEAD_NO_ODOM"
            self.add_event(
                f"Workflow motion rejected: {label} | base not publishing odom "
                f"({self.base_health_text()})"
            )
            return False, "Base not publishing /odom. turtlebot3_node may have crashed."

        ok, msg = self.ensure_motor_power()
        if not ok:
            with self.lock:
                self.status_text = "MOTOR_POWER_ERROR"
            self.add_event(f"Workflow motion rejected: {label} | {msg}")
            return False, msg

        with self.lock:
            if self.motion_active:
                self.add_event(f"Workflow motion rejected: {label} | robot busy")
                return False, "Robot is busy"

            self.motion_active = True
            self.stop_motion_requested = False
            self.status_text = f"WORKFLOW_MOTION_{label}"

        self.add_event(
            f"Workflow motion start: {label} | "
            f"linear_x={linear_x:.3f} | angular_z={angular_z:.3f} | duration={duration:.2f}s"
        )

        ok = True
        fail_reason = None
        start = time.time()

        while time.time() - start < duration:
            with self.lock:
                if self.workflow_stop_requested or self.stop_motion_requested:
                    ok = False
                    fail_reason = "stopped"
                    break

            if not self.base_alive():
                ok = False
                fail_reason = "base_died"
                break

            self.publish_cmd(linear_x, angular_z)
            time.sleep(CMD_PUBLISH_PERIOD)

        self.publish_stop()

        with self.lock:
            self.motion_active = False
            self.stop_motion_requested = False

        if ok:
            self.add_event(f"Workflow motion end: {label}")
            time.sleep(POST_MOVE_SETTLE_SECONDS)
            return True, f"Motion done: {label}"

        if fail_reason == "base_died":
            with self.lock:
                self.status_text = "BASE_DIED_DURING_MOTION"
            self.add_event(
                f"Workflow motion aborted: {label} | base died mid-motion "
                f"({self.base_health_text()})"
            )
            return False, "Base died during motion. turtlebot3_node likely crashed."

        self.add_event(f"Workflow motion stopped: {label}")
        return False, "Motion stopped"

    def turn_by_angle_blocking(self, target_angle_rad, angular_speed, label):
        # Closed-loop rotation using /odom yaw. Drives at angular_speed in the
        # direction implied by sign of target_angle_rad until the robot has
        # rotated by |target_angle_rad|. Far more accurate than time-based
        # rotation because it ignores motor stiction / battery variation.

        if self.cmd_vel_subscriber_count() <= 0:
            with self.lock:
                self.status_text = "NO_CMD_VEL_SUBSCRIBER"
            self.add_event(f"Odom turn rejected: {label} | no /cmd_vel subscriber")
            return False, "No /cmd_vel subscriber"

        if not self.base_alive():
            with self.lock:
                self.status_text = "BASE_DEAD_NO_ODOM"
            self.add_event(f"Odom turn rejected: {label} | base dead")
            return False, "Base not publishing odom"

        with self.lock:
            start_yaw = self.current_yaw

        if start_yaw is None:
            self.add_event(f"Odom turn rejected: {label} | no yaw yet")
            return False, "No yaw available"

        ok, msg = self.ensure_motor_power()
        if not ok:
            with self.lock:
                self.status_text = "MOTOR_POWER_ERROR"
            self.add_event(f"Odom turn rejected: {label} | {msg}")
            return False, msg

        with self.lock:
            if self.motion_active:
                self.add_event(f"Odom turn rejected: {label} | robot busy")
                return False, "Robot busy"
            self.motion_active = True
            self.stop_motion_requested = False
            self.status_text = f"WORKFLOW_TURN_ODOM_{label}"

        target_abs = abs(target_angle_rad)
        direction = 1.0 if target_angle_rad > 0 else -1.0
        angular_z = direction * abs(angular_speed)

        # Stop early to compensate for the robot's continued rotation after
        # we send /cmd_vel = 0. Empirically (from logs) the TurtleBot3 burger
        # at SLOT_TURN_SPEED keeps rotating ~7-10° after stop is published.
        # Setting margin to 7.5° brings post-stop final yaw very close to target.
        stop_margin_rad = 0.13   # ~7.5°
        # Safety timeout: 3x the expected duration
        expected_t = target_abs / max(abs(angular_speed), 0.01)
        timeout_s = max(2.0, expected_t * 3.0)

        self.add_event(
            f"Odom turn start: {label} | "
            f"target={math.degrees(target_angle_rad):+.1f}deg | "
            f"speed={abs(angular_speed):.2f} rad/s | "
            f"start_yaw={math.degrees(start_yaw):+.1f}deg"
        )

        rotated = 0.0
        last_yaw = start_yaw
        start_time = time.time()
        ok = True
        fail_reason = None

        while abs(rotated) < (target_abs - stop_margin_rad):
            with self.lock:
                if self.workflow_stop_requested or self.stop_motion_requested:
                    ok = False
                    fail_reason = "stopped"
                    break
                cur_yaw = self.current_yaw

            if not self.base_alive():
                ok = False
                fail_reason = "base_died"
                break

            if time.time() - start_time > timeout_s:
                ok = False
                fail_reason = "timeout"
                break

            if cur_yaw is None:
                time.sleep(CMD_PUBLISH_PERIOD)
                continue

            delta = shortest_angle_diff(cur_yaw, last_yaw)
            rotated += delta
            last_yaw = cur_yaw

            self.publish_cmd(0.0, angular_z)
            time.sleep(CMD_PUBLISH_PERIOD)

        self.publish_stop()

        # capture final yaw once stopped
        time.sleep(0.15)
        with self.lock:
            final_yaw = self.current_yaw if self.current_yaw is not None else last_yaw

        actual_total = shortest_angle_diff(final_yaw, start_yaw)
        # for >180° turns, sign tracking via shortest_diff is ambiguous; use rotated
        # accumulator if the absolute target exceeded ~170°
        if target_abs > math.radians(170):
            actual_total = rotated

        with self.lock:
            self.motion_active = False
            self.stop_motion_requested = False

        if ok:
            self.add_event(
                f"Odom turn end: {label} | "
                f"target={math.degrees(target_angle_rad):+.1f}deg | "
                f"actual={math.degrees(actual_total):+.1f}deg | "
                f"error={math.degrees(actual_total - target_angle_rad):+.2f}deg"
            )
            time.sleep(POST_MOVE_SETTLE_SECONDS)
            return True, f"Odom turn done: {label}"

        if fail_reason == "base_died":
            with self.lock:
                self.status_text = "BASE_DIED_DURING_MOTION"
            self.add_event(f"Odom turn aborted: {label} | base died")
            return False, "Base died during odom turn"

        if fail_reason == "timeout":
            self.add_event(
                f"Odom turn TIMEOUT: {label} | "
                f"rotated={math.degrees(rotated):+.1f}deg "
                f"of {math.degrees(target_angle_rad):+.1f}deg"
            )
            return False, "Odom turn timeout"

        self.add_event(f"Odom turn stopped: {label}")
        return False, "Odom turn stopped"

    def move_distance_blocking(self, target_distance_m, linear_speed, label):
        # Closed-loop forward (or backward) translation using odom position.
        # target_distance_m is signed: positive=forward, negative=backward.
        # linear_speed is magnitude (m/s); direction comes from sign of target.
        # Tracks Euclidean distance traveled from start point (so small angular
        # drift during a forward move doesn't prematurely stop the motion).

        if self.cmd_vel_subscriber_count() <= 0:
            with self.lock:
                self.status_text = "NO_CMD_VEL_SUBSCRIBER"
            self.add_event(f"Odom move rejected: {label} | no /cmd_vel subscriber")
            return False, "No /cmd_vel subscriber"

        if not self.base_alive():
            with self.lock:
                self.status_text = "BASE_DEAD_NO_ODOM"
            self.add_event(f"Odom move rejected: {label} | base dead")
            return False, "Base not publishing odom"

        with self.lock:
            start_x = self.current_pos_x
            start_y = self.current_pos_y

        if start_x is None or start_y is None:
            self.add_event(f"Odom move rejected: {label} | no position yet")
            return False, "No odom position available"

        ok, msg = self.ensure_motor_power()
        if not ok:
            with self.lock:
                self.status_text = "MOTOR_POWER_ERROR"
            self.add_event(f"Odom move rejected: {label} | {msg}")
            return False, msg

        with self.lock:
            if self.motion_active:
                self.add_event(f"Odom move rejected: {label} | robot busy")
                return False, "Robot busy"
            self.motion_active = True
            self.stop_motion_requested = False
            self.status_text = f"WORKFLOW_MOVE_ODOM_{label}"

        target_abs = abs(float(target_distance_m))
        direction = 1.0 if target_distance_m >= 0 else -1.0
        linear_x = direction * abs(float(linear_speed))

        # Stop early to compensate for translation overshoot after stop.
        # Empirically (from logs) the robot continues ~3 cm past the stop
        # point at SLOT_FORWARD_SPEED.
        stop_margin_m = 0.025  # 2.5 cm
        expected_t = target_abs / max(abs(linear_speed), 0.005)
        timeout_s = max(2.0, expected_t * 3.0)

        self.add_event(
            f"Odom move start: {label} | "
            f"target={target_distance_m:+.3f}m | speed={abs(linear_speed):.3f} m/s | "
            f"start=({start_x:.3f}, {start_y:.3f})"
        )

        traveled = 0.0
        start_time = time.time()
        ok = True
        fail_reason = None

        while traveled < (target_abs - stop_margin_m):
            with self.lock:
                if self.workflow_stop_requested or self.stop_motion_requested:
                    ok = False
                    fail_reason = "stopped"
                    break
                cur_x = self.current_pos_x
                cur_y = self.current_pos_y

            if not self.base_alive():
                ok = False
                fail_reason = "base_died"
                break

            if time.time() - start_time > timeout_s:
                ok = False
                fail_reason = "timeout"
                break

            if cur_x is None:
                time.sleep(CMD_PUBLISH_PERIOD)
                continue

            dx = cur_x - start_x
            dy = cur_y - start_y
            traveled = math.sqrt(dx * dx + dy * dy)

            self.publish_cmd(linear_x, 0.0)
            time.sleep(CMD_PUBLISH_PERIOD)

        self.publish_stop()
        time.sleep(0.15)

        with self.lock:
            final_x = self.current_pos_x if self.current_pos_x is not None else start_x
            final_y = self.current_pos_y if self.current_pos_y is not None else start_y

        actual_traveled = math.sqrt(
            (final_x - start_x) ** 2 + (final_y - start_y) ** 2
        )

        with self.lock:
            self.motion_active = False
            self.stop_motion_requested = False

        if ok:
            self.add_event(
                f"Odom move end: {label} | target={target_distance_m:+.3f}m | "
                f"traveled={direction*actual_traveled:+.3f}m | "
                f"error={direction*actual_traveled - target_distance_m:+.3f}m"
            )
            time.sleep(POST_MOVE_SETTLE_SECONDS)
            return True, f"Odom move done: {label}"

        if fail_reason == "base_died":
            with self.lock:
                self.status_text = "BASE_DIED_DURING_MOTION"
            self.add_event(f"Odom move aborted: {label} | base died")
            return False, "Base died during odom move"

        if fail_reason == "timeout":
            self.add_event(
                f"Odom move TIMEOUT: {label} | "
                f"traveled={actual_traveled:.3f}m of {target_abs:.3f}m"
            )
            return False, "Odom move timeout"

        self.add_event(f"Odom move stopped: {label}")
        return False, "Odom move stopped"

    def wait_for_fresh_frame(self, max_wait_s=1.5, count=1, min_settle_s=0.0):
        # Wait until `count` camera frames have arrived AFTER this call
        # started, optionally after a minimum settle sleep first.
        #
        # Why count > 1: a single "fresh" frame after a motion may still be
        # motion-blurred because camera_ros may have queued frames captured
        # during the motion. Waiting for several consecutive fresh frames
        # gives the camera/scene real time to stabilize before we read
        # latest_bbox.
        if min_settle_s > 0:
            time.sleep(min_settle_s)

        entry_time = time.time()
        seen = 0
        last_seen_time = None
        deadline = entry_time + max_wait_s

        while time.time() < deadline:
            with self.lock:
                lt = self.last_frame_time
                stop = self.workflow_stop_requested or self.stop_motion_requested
            if stop:
                return False
            if lt is not None and lt > entry_time and lt != last_seen_time:
                seen += 1
                last_seen_time = lt
                if seen >= count:
                    return True
            time.sleep(0.05)
        return False

    def vision_align_to_card(self, slot_id, title):
        # Visual fine-tune AFTER the closed-loop motion lands the robot
        # approximately in front of a slot. Reads self.latest_bbox (produced
        # by the existing detect_best_card pipeline — NOT modified here) and
        # nudges the robot until the card is centered horizontally AND the
        # card height in frame matches the configured target ratio.
        #
        # Handles three correction axes:
        #   (1) X centering   -> small in-place rotation
        #   (2) Distance      -> small forward / backward move
        #   (3) Card not seen -> short forward search (up to N small steps)
        #
        # Returns (ok, message). Never modifies detection logic.
        if not VISION_ALIGN_ENABLED:
            return True, "vision align disabled"

        # CRITICAL: wait for at least 2 consecutive fresh frames AFTER a
        # short settle. This guarantees the camera has time to flush any
        # motion-blurred queued frames and produce a sharp current frame
        # before we read latest_bbox. Without this, slots that need 0
        # alignment steps (vision_align proceeds straight to save) capture
        # a blurry image because there's no organic settle from corrective
        # motions.
        self.wait_for_fresh_frame(max_wait_s=1.5, count=2, min_settle_s=0.30)

        # Track cumulative forward motion to prevent driving INTO the card.
        # If step_to_next undershoots, search-forward + correction-forward
        # could otherwise stack to 20+ cm and the saved reference becomes
        # an extreme close-up (see slot 3 from the 19:39 test where ref was
        # captured from inside the card showing only the central pip).
        total_forward_m = 0.0

        # Phase A: search if no card visible
        with self.lock:
            bbox = self.latest_bbox

        if bbox is None:
            self.trace("VALIGN", f"Slot {slot_id} {title}: no card visible, searching forward")
            search_ok = False
            for sstep in range(VISION_ALIGN_SEARCH_MAX_STEPS):
                with self.lock:
                    if self.workflow_stop_requested:
                        return False, "Workflow stopped"

                if (total_forward_m + VISION_ALIGN_SEARCH_FORWARD_M
                        > VISION_ALIGN_MAX_TOTAL_FORWARD_M):
                    self.add_event(
                        f"Vision align {title} Slot {slot_id}: search aborted, "
                        f"would exceed max total forward "
                        f"{VISION_ALIGN_MAX_TOTAL_FORWARD_M * 100:.0f}cm"
                    )
                    break

                ok, _ = self.move_distance_blocking(
                    VISION_ALIGN_SEARCH_FORWARD_M,
                    VISION_ALIGN_LINEAR_SPEED,
                    f"VALIGN_SEARCH_SLOT_{slot_id}_{title}_STEP_{sstep + 1}"
                )
                if not ok:
                    return False, "Search forward failed"
                total_forward_m += VISION_ALIGN_SEARCH_FORWARD_M

                # Wait briefly for fresh detection
                wait_t = time.time()
                while time.time() - wait_t < 1.0:
                    with self.lock:
                        if self.latest_bbox is not None:
                            search_ok = True
                            break
                    time.sleep(0.1)
                if search_ok:
                    self.add_event(
                        f"Vision align {title} Slot {slot_id}: card found after "
                        f"{(sstep + 1) * VISION_ALIGN_SEARCH_FORWARD_M * 100:.0f}cm forward search"
                    )
                    break

            if not search_ok:
                self.add_event(f"Vision align {title} Slot {slot_id}: card NOT FOUND after search")
                return False, "NO_CARD_AFTER_SEARCH"

        # Phase B: iterative alignment (centering + distance)
        for step in range(VISION_ALIGN_MAX_STEPS):
            with self.lock:
                if self.workflow_stop_requested:
                    return False, "Workflow stopped"

                raw = self.latest_raw.copy() if self.latest_raw is not None else None
                bbox = self.latest_bbox

            if raw is None:
                time.sleep(0.1)
                continue

            if bbox is None:
                # Card disappeared during alignment — rare. Try one search step
                # but ONLY if we still have forward budget.
                self.trace("VALIGN", f"Slot {slot_id} {title}: bbox lost mid-align step {step + 1}")
                if (total_forward_m + VISION_ALIGN_SEARCH_FORWARD_M
                        > VISION_ALIGN_MAX_TOTAL_FORWARD_M):
                    self.add_event(
                        f"Vision align {title} Slot {slot_id}: bbox lost and no "
                        f"forward budget left — aborting"
                    )
                    return False, "BBOX_LOST_AT_FORWARD_LIMIT"
                self.move_distance_blocking(
                    VISION_ALIGN_SEARCH_FORWARD_M,
                    VISION_ALIGN_LINEAR_SPEED,
                    f"VALIGN_RECOVER_SLOT_{slot_id}_{title}"
                )
                total_forward_m += VISION_ALIGN_SEARCH_FORWARD_M
                time.sleep(0.5)
                continue

            frame_h, frame_w = raw.shape[:2]
            x, y, bw, bh = bbox
            card_cx = x + bw / 2.0
            frame_cx = frame_w / 2.0
            x_error = card_cx - frame_cx
            x_tol = max(35.0, frame_w * VISION_ALIGN_X_TOL_RATIO)

            target_h = frame_h * VISION_ALIGN_TARGET_HEIGHT_RATIO
            h_tol = frame_h * VISION_ALIGN_HEIGHT_TOL_RATIO
            h_error = bh - target_h  # positive = too close, negative = too far

            x_ok = abs(x_error) <= x_tol
            h_ok = abs(h_error) <= h_tol

            if x_ok and h_ok:
                self.add_event(
                    f"Vision align {title} Slot {slot_id}: aligned in {step} steps | "
                    f"x_err={x_error:.1f}px (tol {x_tol:.0f}) | "
                    f"h_err={h_error:.1f}px (tol {h_tol:.0f})"
                )
                return True, "Aligned"

            # Decide which axis to correct first: centering takes priority
            # (because forward/back motion changes both axes, but turn only
            # changes x).
            if not x_ok:
                error_ratio = min(1.0, abs(x_error) / (frame_w / 2.0))
                duration = (
                    VISION_ALIGN_TURN_DURATION_MIN
                    + (VISION_ALIGN_TURN_DURATION_MAX - VISION_ALIGN_TURN_DURATION_MIN)
                    * error_ratio
                )
                angular_z = -VISION_ALIGN_TURN_SPEED if x_error > 0 else +VISION_ALIGN_TURN_SPEED
                self.trace(
                    "VALIGN",
                    f"Slot {slot_id} {title} step {step + 1}: x_err={x_error:.1f}px "
                    f"-> turn {angular_z:+.2f} rad/s for {duration:.2f}s"
                )
                ok, msg = self.run_motion_blocking(
                    0.0, angular_z, duration,
                    f"VALIGN_TURN_SLOT_{slot_id}_{title}_STEP_{step + 1}"
                )
                if not ok:
                    return False, msg
                continue

            # x is OK but distance isn't — adjust forward/backward
            if h_error < 0:
                # Card too small -> move forward (subject to total-forward cap)
                step_m = VISION_ALIGN_FORWARD_STEP_M
                if (total_forward_m + step_m) > VISION_ALIGN_MAX_TOTAL_FORWARD_M:
                    self.add_event(
                        f"Vision align {title} Slot {slot_id}: forward correction "
                        f"capped at {VISION_ALIGN_MAX_TOTAL_FORWARD_M * 100:.0f}cm "
                        f"total — accepting current alignment"
                    )
                    break
                self.trace(
                    "VALIGN",
                    f"Slot {slot_id} {title} step {step + 1}: h_err={h_error:.1f}px "
                    f"-> forward {step_m * 100:.0f}cm "
                    f"(total fwd so far {total_forward_m * 100:.1f}cm)"
                )
                ok, msg = self.move_distance_blocking(
                    +step_m, VISION_ALIGN_LINEAR_SPEED,
                    f"VALIGN_FWD_SLOT_{slot_id}_{title}_STEP_{step + 1}"
                )
                if ok:
                    total_forward_m += step_m
            else:
                # Card too big -> move backward (small step, safer)
                step_m = VISION_ALIGN_BACKWARD_STEP_M
                self.trace(
                    "VALIGN",
                    f"Slot {slot_id} {title} step {step + 1}: h_err={h_error:.1f}px "
                    f"-> backward {step_m * 100:.0f}cm"
                )
                ok, msg = self.move_distance_blocking(
                    -step_m, VISION_ALIGN_LINEAR_SPEED,
                    f"VALIGN_BACK_SLOT_{slot_id}_{title}_STEP_{step + 1}"
                )
                if ok:
                    total_forward_m -= step_m  # reduces forward count
            if not ok:
                return False, msg

        self.add_event(
            f"Vision align {title} Slot {slot_id}: max steps reached, proceeding anyway"
        )
        return True, "Max align steps reached"

    def auto_center_current_card(self, slot_id, title):
        if not AUTO_CENTER_BEFORE_CAPTURE:
            return True, "Auto center disabled"

        # Wait briefly for a fresh detection before giving up.
        wait_start = time.time()
        while time.time() - wait_start < CARD_WAIT_TIMEOUT:
            with self.lock:
                if self.workflow_stop_requested:
                    return False, "Workflow stopped"
                bbox_ready = self.latest_bbox is not None
            if bbox_ready:
                break
            time.sleep(0.1)

        step = 0
        while step < AUTO_CENTER_MAX_STEPS:
            with self.lock:
                if self.workflow_stop_requested:
                    return False, "Workflow stopped"

                raw = self.latest_raw.copy() if self.latest_raw is not None else None
                bbox = self.latest_bbox

            if raw is None or bbox is None:
                # Don't burn an iteration when there's no detection — wait for one.
                self.trace(
                    "AUTOCENTER",
                    f"Slot {slot_id} {title}: no bbox yet, waiting"
                )
                wait_more = time.time()
                while time.time() - wait_more < 1.5:
                    with self.lock:
                        if self.workflow_stop_requested:
                            return False, "Workflow stopped"
                        if self.latest_bbox is not None:
                            break
                    time.sleep(0.1)
                with self.lock:
                    if self.latest_bbox is None:
                        self.add_event(
                            f"Auto center {title} Slot {slot_id}: no card detected after wait"
                        )
                        return False, "NO_CARD_DETECTED"
                continue

            frame_h, frame_w = raw.shape[:2]
            x, y, bw, bh = bbox
            card_cx = x + bw / 2.0
            frame_cx = frame_w / 2.0
            error = card_cx - frame_cx
            tolerance = max(35.0, frame_w * AUTO_CENTER_TOLERANCE_RATIO)

            if abs(error) <= tolerance:
                self.add_event(
                    f"Auto center {title} Slot {slot_id}: centered | "
                    f"error={error:.1f}px | tolerance={tolerance:.1f}px"
                )
                return True, "Card centered"

            # Proportional step: larger error -> longer turn, smaller error -> shorter
            # error_ratio: 0..1 (1 means card is at the frame edge)
            error_ratio = min(1.0, abs(error) / (frame_w / 2.0))
            duration = (
                AUTO_CENTER_TURN_DURATION_MIN
                + (AUTO_CENTER_TURN_DURATION_MAX - AUTO_CENTER_TURN_DURATION_MIN)
                * error_ratio
            )
            angular_z = -AUTO_CENTER_TURN_SPEED if error > 0 else +AUTO_CENTER_TURN_SPEED

            step += 1
            self.trace(
                "AUTOCENTER",
                f"Slot {slot_id} {title} step {step}: error={error:.1f}px "
                f"(ratio={error_ratio:.2f}) -> duration={duration:.2f}s "
                f"angular_z={angular_z:+.2f}"
            )
            ok, msg = self.run_motion_blocking(
                0.0,
                angular_z,
                duration,
                f"AUTO_CENTER_{title}_SLOT_{slot_id}_STEP_{step}"
            )
            if not ok:
                return False, msg

        self.add_event(f"Auto center {title} Slot {slot_id}: max steps reached")
        return True, "Auto center max steps reached"

    def step_to_next_slot_workflow(self, label, distance_m=None):
        # Fully closed-loop: turn LEFT 90°, forward N cm (default
        # SLOT_FORWARD_DISTANCE_M, but caller may override with the actual
        # measured slot-to-slot gap), turn RIGHT 90°. Vision alignment is
        # done by the caller AFTER this step lands.
        if distance_m is None:
            distance_m = SLOT_FORWARD_DISTANCE_M

        ok, msg = self.turn_by_angle_blocking(
            +math.pi / 2.0,
            SLOT_TURN_SPEED,
            f"{label}_TURN_LEFT_90"
        )
        if not ok:
            return False, msg

        ok, msg = self.move_distance_blocking(
            +float(distance_m),
            SLOT_FORWARD_SPEED,
            f"{label}_FORWARD_SLOT_GAP"
        )
        if not ok:
            return False, msg

        return self.turn_by_angle_blocking(
            -math.pi / 2.0,
            SLOT_TURN_SPEED,
            f"{label}_TURN_RIGHT_90"
        )

    def step_to_previous_slot_workflow(self, label, distance_m=None):
        if distance_m is None:
            distance_m = SLOT_FORWARD_DISTANCE_M

        ok, msg = self.turn_by_angle_blocking(
            -math.pi / 2.0,
            SLOT_TURN_SPEED,
            f"{label}_TURN_RIGHT_90"
        )
        if not ok:
            return False, msg

        ok, msg = self.move_distance_blocking(
            +float(distance_m),
            SLOT_FORWARD_SPEED,
            f"{label}_FORWARD_SLOT_GAP"
        )
        if not ok:
            return False, msg

        return self.turn_by_angle_blocking(
            +math.pi / 2.0,
            SLOT_TURN_SPEED,
            f"{label}_TURN_LEFT_90"
        )

    def turn_around_workflow(self, label):
        return self.turn_by_angle_blocking(
            +math.pi,
            SLOT_TURN_SPEED,
            label
        )

    def scan_all_slots_workflow(self, mode, reverse=False):
        if mode not in ("save", "check"):
            return False, f"Unknown scan mode: {mode}"

        title = "SAVE" if mode == "save" else "CHECK"
        slot_ids = list(range(1, NUM_SLOTS + 1))

        if reverse:
            slot_ids.reverse()

        self.trace("SCAN", f"scan_all_slots_workflow start | mode={mode} | reverse={reverse} | order={slot_ids}")

        for index, slot_id in enumerate(slot_ids):
            self.trace("SCAN", f"--- Slot {slot_id} ({title}) iteration {index + 1}/{len(slot_ids)} ---")
            with self.lock:
                if self.workflow_stop_requested:
                    self.trace("SCAN", f"Slot {slot_id}: stop requested before centering")
                    return False, "Workflow stopped"
                self.status_text = f"WORKFLOW_{title}_SLOT_{slot_id}_CENTERING"

            self.trace("SCAN", f"Slot {slot_id}: auto_center_current_card start")
            ok, msg = self.auto_center_current_card(slot_id, title)
            self.trace("SCAN", f"Slot {slot_id}: auto_center_current_card end | ok={ok} | msg={msg}")
            if not ok:
                return False, msg

            with self.lock:
                if self.workflow_stop_requested:
                    self.trace("SCAN", f"Slot {slot_id}: stop requested before wait_for_stable_card")
                    return False, "Workflow stopped"
                self.status_text = f"WORKFLOW_{title}_SLOT_{slot_id}_WAITING_CARD"

            self.trace("SCAN", f"Slot {slot_id}: wait_for_stable_card start")
            ok, msg = self.wait_for_stable_card()
            self.trace("SCAN", f"Slot {slot_id}: wait_for_stable_card end | ok={ok} | msg={msg}")
            if not ok:
                self.add_event(f"Workflow {title} Slot {slot_id}: FAILED | {msg}")
                with self.lock:
                    self.status_text = f"WORKFLOW_{title}_SLOT_{slot_id}_{msg}"
                return False, msg

            self.trace("SCAN", f"Slot {slot_id}: {mode}_slot start")
            if mode == "save":
                ok, msg = self.save_slot(slot_id)
            else:
                ok, msg = self.check_slot(slot_id)
            self.trace("SCAN", f"Slot {slot_id}: {mode}_slot end | ok={ok} | msg={msg}")

            if not ok:
                self.add_event(f"Workflow {title} Slot {slot_id}: FAILED | {msg}")
                return False, msg

            if index < len(slot_ids) - 1:
                step_label = f"STEP_PREVIOUS_AFTER_SLOT_{slot_id}" if reverse else f"STEP_NEXT_AFTER_SLOT_{slot_id}"
                self.trace("SCAN", f"Slot {slot_id}: stepping to next | label={step_label}")
                if reverse:
                    ok, msg = self.step_to_previous_slot_workflow(step_label)
                else:
                    ok, msg = self.step_to_next_slot_workflow(step_label)
                self.trace("SCAN", f"Slot {slot_id}: step done | ok={ok} | msg={msg}")
                if not ok:
                    return False, msg

        self.update_result_text()
        self.trace("SCAN", f"scan_all_slots_workflow done | mode={mode}")
        return True, f"Auto {mode} all slots done"

    def return_to_slot_1_workflow(self):
        for step in range(NUM_SLOTS - 1):
            with self.lock:
                if self.workflow_stop_requested:
                    return False, "Workflow stopped"

            ok, msg = self.step_to_previous_slot_workflow(
                f"RETURN_TO_SLOT_1_STEP_{step + 1}"
            )
            if not ok:
                return False, msg

        return True, "Returned to Slot 1"

    def workflow_worker(self, workflow_name):
        ok = True
        message = "OK"

        self.trace("WORKFLOW", f"Worker thread entered: {workflow_name}")
        self.add_event(f"Workflow start: {workflow_name}")

        try:
            if workflow_name == "FULL_AUTO_ONE_SCAN":
                ok, message = self.run_full_auto_one_scan_workflow()

            elif workflow_name == "AUTO_SAVE_ALL":
                ok, message = self.scan_all_slots_workflow("save")

            elif workflow_name == "AUTO_CHECK_ALL":
                ok, message = self.scan_all_slots_workflow("check")

            elif workflow_name == "AUTO_RETURN_TO_SLOT_1":
                ok, message = self.return_to_slot_1_workflow()

            elif workflow_name == "STEP_NEXT_SLOT":
                ok, message = self.step_to_next_slot_workflow("MANUAL_STEP_NEXT_SLOT")

            elif workflow_name == "STEP_PREVIOUS_SLOT":
                ok, message = self.step_to_previous_slot_workflow("MANUAL_STEP_PREVIOUS_SLOT")

            elif workflow_name == "AUTO_LOOK_AWAY_RIGHT":
                ok, message = self.turn_around_workflow("LOOK_AWAY_TURN_180")

            elif workflow_name == "AUTO_LOOK_BACK_LEFT":
                ok, message = self.turn_around_workflow("LOOK_BACK_TURN_180")

            elif workflow_name == "START_SCAN":
                # Full magic-trick flow in one shot:
                #   1. Save references for slots 1..5 (forward scan)
                #   2. Turn 180° away from cards
                #   3. Wait DEMO_WAIT_SECONDS (user flips one card)
                #   4. Turn 180° back to face card 5
                #   5. Check all 5 slots in reverse (5..1)
                #   6. Save report; result text shows which slot rotated
                self.add_event("START_SCAN phase 1/5: saving references (slot 1 -> 5)")
                ok, message = self.scan_all_slots_workflow("save")

                if ok:
                    self.add_event("START_SCAN phase 2/5: turning 180 degrees away from cards")
                    ok, message = self.turn_around_workflow("LOOK_AWAY_TURN_180")

                if ok:
                    self.add_event(
                        f"START_SCAN phase 3/5: waiting {DEMO_WAIT_SECONDS:.0f}s "
                        f"for card rotation"
                    )
                    ok, message = self.wait_with_stop_check(DEMO_WAIT_SECONDS, "CARD_ROTATION")

                if ok:
                    self.add_event("START_SCAN phase 4/5: turning 180 degrees back to face slot 5")
                    ok, message = self.turn_around_workflow("LOOK_BACK_TURN_180")

                if ok:
                    self.add_event("START_SCAN phase 5/5: checking references (slot 5 -> 1)")
                    ok, message = self.scan_all_slots_workflow("check", reverse=True)

                if ok:
                    report_path = self.save_report()
                    message = f"Magic trick complete. Report: {report_path}"

            elif workflow_name == "AUTO_RUN":
                # Fully autonomous run with closed-loop motion + vision align:
                #   Phase 1: at slot 1, scan -> step+align to slot 2 -> scan -> ... -> scan slot 5
                #   Phase 2: turn 90° AWAY from cards (not 180° — less drift)
                #   Phase 3: wait DEMO_WAIT_SECONDS for user to flip a card
                #   Phase 4: turn 90° BACK to face slot 5
                #   Phase 5: at slot 5, scan(check) -> step+align to slot 4 -> ... -> scan slot 1
                #   Phase 6: report
                with self.lock:
                    self.scan_phase = "SAVE_PHASE"
                    self.cross_match_result = None

                ok, message = self._auto_run_save_phase()
                if ok:
                    self.add_event("AUTO_RUN: turning 90 degrees away from cards")
                    ok, message = self.turn_by_angle_blocking(
                        +math.pi / 2.0,
                        SLOT_TURN_SPEED,
                        "AUTO_LOOK_AWAY_90"
                    )
                if ok:
                    with self.lock:
                        self.scan_phase = "WAIT_PHASE"
                        self.wait_started_at = time.time()
                    self.add_event(
                        f"AUTO_RUN: waiting {DEMO_WAIT_SECONDS:.0f}s for card rotation"
                    )
                    ok, message = self.wait_with_stop_check(DEMO_WAIT_SECONDS, "AUTO_CARD_ROTATION")
                    with self.lock:
                        self.wait_started_at = None
                if ok:
                    self.add_event("AUTO_RUN: turning 90 degrees back to face slot 5")
                    ok, message = self.turn_by_angle_blocking(
                        -math.pi / 2.0,
                        SLOT_TURN_SPEED,
                        "AUTO_LOOK_BACK_90"
                    )
                if ok:
                    with self.lock:
                        self.scan_phase = "CHECK_PHASE"
                    ok, message = self._auto_run_check_phase()
                if ok:
                    self.cross_match_result = self.cross_match_analysis()
                    self.update_result_text()
                    report_path = self.save_report()
                    with self.lock:
                        self.scan_phase = "DONE_PHASE"
                        self.last_report_path = report_path
                    message = f"Auto run complete. Report: {report_path}"

            else:
                ok = False
                message = f"Unknown workflow: {workflow_name}"

        except Exception as e:
            ok = False
            message = f"Workflow error: {workflow_name}: {e}"
            print("[WORKFLOW ERROR]", message, flush=True)
            self.trace_exc("WORKFLOW_ERR", f"Worker {workflow_name}")

        finally:
            self.publish_stop()

            with self.lock:
                self.workflow_active = False
                self.workflow_stop_requested = False
                if ok:
                    self.status_text = f"WORKFLOW_{workflow_name}_DONE"
                else:
                    self.status_text = f"WORKFLOW_{workflow_name}_FAILED"
                    self.ui_phase = "failed"

            self.add_event(f"Workflow end: {workflow_name} | ok={ok} | {message}")
            self.trace("WORKFLOW", f"Worker thread exited: {workflow_name} | ok={ok} | {message}")
            self.update_result_text()

    def start_workflow(self, workflow_name):
        with self.lock:
            if self.workflow_active:
                self.status_text = "WORKFLOW_BUSY"
                self.add_event(f"Workflow rejected: {workflow_name} | workflow busy")
                return False, "Workflow is already running"

            if self.motion_active:
                self.status_text = "MOTION_BUSY"
                self.add_event(f"Workflow rejected: {workflow_name} | robot moving")
                return False, "Robot is moving"

            self.workflow_active = True
            self.workflow_stop_requested = False
            self.stop_motion_requested = False
            self.ui_phase = "starting"
            self.status_text = f"WORKFLOW_{workflow_name}_STARTED"

        th = threading.Thread(
            target=self.workflow_worker,
            args=(workflow_name,),
            daemon=True
        )
        th.start()

        return True, f"Workflow started: {workflow_name}"

    # ========================================================
    # MOTION
    # ========================================================

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def cmd_vel_subscriber_count(self):
        try:
            return int(self.cmd_pub.get_subscription_count())
        except Exception:
            return 0

    def odom_callback(self, msg):
        now = time.time()
        try:
            yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        except Exception:
            yaw = None

        try:
            pos_x = float(msg.pose.pose.position.x)
            pos_y = float(msg.pose.pose.position.y)
        except Exception:
            pos_x = pos_y = None

        with self.lock:
            previous_last = self.base_last_odom_time
            self.base_last_odom_time = now
            self.base_odom_count += 1
            if yaw is not None:
                self.current_yaw = yaw
            if pos_x is not None:
                self.current_pos_x = pos_x
                self.current_pos_y = pos_y
            previous_state = self.last_base_alive_state

        if previous_last is None:
            self.trace("BASE", f"First odom received (count=1)")
            with self.lock:
                self.last_base_alive_state = True
        elif (now - previous_last) > BASE_ALIVE_TIMEOUT:
            gap = now - previous_last
            self.trace("BASE", f"Odom RESUMED after gap of {gap:.2f}s (was DEAD)")
            with self.lock:
                self.last_base_alive_state = True
        elif previous_state is False:
            self.trace("BASE", f"Odom RESUMED (was DEAD)")
            with self.lock:
                self.last_base_alive_state = True

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

        with self.lock:
            self.battery_percent = percent
            self.battery_state = state
            self.battery_last_time = time.time()

    def battery_status(self):
        with self.lock:
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

    def base_alive(self):
        with self.lock:
            last = self.base_last_odom_time
        if last is None:
            return False
        return (time.time() - last) <= BASE_ALIVE_TIMEOUT

    def base_health_text(self):
        with self.lock:
            last = self.base_last_odom_time
            count = self.base_odom_count
        if last is None:
            return "NO_ODOM_YET"
        age = time.time() - last
        if age <= BASE_ALIVE_TIMEOUT:
            return f"ALIVE (odom age {age:.2f}s, count {count})"
        return f"DEAD (no odom for {age:.1f}s, last count {count})"

    def ensure_motor_power(self):
        if not self.motor_power_client.service_is_ready():
            if not self.motor_power_client.wait_for_service(timeout_sec=0.5):
                self.add_event("Motor power service not available")
                return False, "Motor power service not available"

        request = SetBool.Request()
        request.data = True

        future = self.motor_power_client.call_async(request)
        start = time.time()

        while not future.done() and time.time() - start < 2.0:
            time.sleep(0.05)

        if not future.done():
            self.add_event("Motor power enable timed out")
            return False, "Motor power enable timed out"

        try:
            response = future.result()
        except Exception as e:
            self.add_event(f"Motor power enable failed: {e}")
            return False, f"Motor power enable failed: {e}"

        if not response.success:
            self.add_event(f"Motor power rejected: {response.message}")
            return False, f"Motor power rejected: {response.message}"

        return True, "Motor power enabled"

    def publish_stop(self):
        for _ in range(3):
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.04)

    def motion_worker(self, linear_x, angular_z, duration, label):
        if self.cmd_vel_subscriber_count() <= 0:
            with self.lock:
                self.motion_active = False
                self.status_text = "NO_CMD_VEL_SUBSCRIBER"
            self.add_event(f"Motion rejected: {label} | no /cmd_vel subscriber")
            return

        if not self.base_alive():
            with self.lock:
                self.motion_active = False
                self.status_text = "BASE_DEAD_NO_ODOM"
            self.add_event(
                f"Motion rejected: {label} | base not publishing odom "
                f"({self.base_health_text()})"
            )
            return

        ok, msg = self.ensure_motor_power()
        if not ok:
            with self.lock:
                self.motion_active = False
                self.status_text = "MOTOR_POWER_ERROR"
            self.add_event(f"Motion rejected: {label} | {msg}")
            return

        with self.lock:
            self.motion_active = True
            self.stop_motion_requested = False
            self.status_text = f"MOTION_{label}"

        self.add_event(
            f"Motion start: {label} | "
            f"linear_x={linear_x:.3f} | angular_z={angular_z:.3f} | duration={duration:.2f}s"
        )

        start = time.time()
        base_died = False

        while time.time() - start < duration:
            with self.lock:
                if self.stop_motion_requested:
                    break

            if not self.base_alive():
                base_died = True
                break

            self.publish_cmd(linear_x, angular_z)
            time.sleep(CMD_PUBLISH_PERIOD)

        self.publish_stop()

        with self.lock:
            self.motion_active = False
            self.stop_motion_requested = False
            if base_died:
                self.status_text = "BASE_DIED_DURING_MOTION"
            else:
                self.status_text = "READY"

        if base_died:
            self.add_event(
                f"Motion aborted: {label} | base died mid-motion "
                f"({self.base_health_text()})"
            )
        else:
            self.add_event(f"Motion end: {label}")

    def start_motion(self, linear_x, angular_z, duration, label):
        with self.lock:
            if self.workflow_active:
                self.status_text = "WORKFLOW_BUSY"
                self.add_event(f"Motion rejected: {label} | workflow busy")
                return False, "Workflow is running"

            if self.motion_active:
                self.status_text = "MOTION_BUSY"
                self.add_event(f"Motion rejected: {label} | robot busy")
                return False, "Robot is busy"

            self.stop_motion_requested = False

        if self.cmd_vel_subscriber_count() <= 0:
            with self.lock:
                self.status_text = "NO_CMD_VEL_SUBSCRIBER"
            self.add_event(f"Motion rejected: {label} | no /cmd_vel subscriber")
            return False, "No /cmd_vel subscriber. Is turtlebot3_bringup running?"

        if not self.base_alive():
            with self.lock:
                self.status_text = "BASE_DEAD_NO_ODOM"
            self.add_event(
                f"Motion rejected: {label} | base not publishing odom "
                f"({self.base_health_text()})"
            )
            return False, "Base not publishing /odom. turtlebot3_node may have crashed."

        th = threading.Thread(
            target=self.motion_worker,
            args=(linear_x, angular_z, duration, label),
            daemon=True
        )
        th.start()

        return True, f"Motion started: {label}"

    def stop_robot(self):
        with self.lock:
            self.stop_motion_requested = True
            self.workflow_stop_requested = True
            self.status_text = "STOP_REQUESTED"

        self.publish_stop()
        time.sleep(0.1)
        self.publish_stop()

        self.add_event("Stop Robot pressed")

        report_path = self.save_report()

        with self.lock:
            self.status_text = "STOPPED"
            self.ui_phase = "stopped"
            self.last_report_path = report_path

        return True, f"Robot stopped. Report saved: {report_path}"

    def save_debug_snapshot(self):
        ensure_dirs()

        stamp = now_file_str()
        prefix = os.path.join(LOG_DIR, f"debug_snapshot_{stamp}")

        with self.lock:
            raw = self.latest_raw.copy() if self.latest_raw is not None else None
            mask = self.latest_mask.copy() if self.latest_mask is not None else None
            crop_raw = self.latest_crop_raw.copy() if self.latest_crop_raw is not None else None
            crop_prepared = self.latest_crop_prepared.copy() if self.latest_crop_prepared is not None else None
            bbox = self.latest_bbox
            card_found = self.latest_card_found
            status_text = self.status_text
            result_text = self.result_text
            workflow_active = self.workflow_active
            motion_active = self.motion_active
            cmd_vel_subscribers = self.cmd_vel_subscriber_count()
            refs_saved = {
                i: self.references[i] is not None
                for i in range(1, NUM_SLOTS + 1)
            }
            checks_copy = {
                i: self.checks[i].copy() if self.checks[i] is not None else None
                for i in range(1, NUM_SLOTS + 1)
            }
            events_tail = list(self.event_log[-25:])

        paths = {}

        if raw is not None:
            raw_path = prefix + "_raw.jpg"
            cv2.imwrite(raw_path, raw)
            paths["raw"] = raw_path

        if mask is not None:
            mask_path = prefix + "_mask.jpg"
            cv2.imwrite(mask_path, mask)
            paths["mask"] = mask_path

        if crop_raw is not None:
            crop_path = prefix + "_crop.jpg"
            cv2.imwrite(crop_path, crop_raw)
            paths["crop"] = crop_path

        if crop_prepared is not None:
            prepared_path = prefix + "_prepared.jpg"
            cv2.imwrite(prepared_path, crop_prepared)
            paths["prepared"] = prepared_path

        display_path = prefix + "_display.jpg"
        display = self.build_display_frame()
        cv2.imwrite(display_path, display)
        paths["display"] = display_path

        txt_path = prefix + ".txt"
        lines = []
        lines.append("========================================")
        lines.append("Debug Snapshot")
        lines.append("========================================")
        lines.append(f"Created: {now_str()}")
        lines.append(f"Status: {status_text}")
        lines.append(f"Result: {result_text}")
        lines.append(f"Card found: {card_found}")
        lines.append(f"BBox: {bbox}")
        lines.append(f"Workflow active: {workflow_active}")
        lines.append(f"Motion active: {motion_active}")
        lines.append(f"Cmd_vel subscribers: {cmd_vel_subscribers}")
        lines.append(f"Base health: {self.base_health_text()}")
        lines.append("")

        if raw is None:
            lines.append("Raw frame: NONE")
        else:
            lines.append(f"Raw frame shape: {raw.shape}")

        if mask is None:
            lines.append("Mask: NONE")
        else:
            lines.append(f"Mask shape: {mask.shape}")
            lines.append(f"Mask nonzero ratio: {float(np.mean(mask > 0)):.4f}")

        if crop_raw is None:
            lines.append("Crop raw: NONE")
        else:
            lines.append(f"Crop raw shape: {crop_raw.shape}")

        if crop_prepared is None:
            lines.append("Crop prepared: NONE")
        else:
            lines.append(f"Crop prepared shape: {crop_prepared.shape}")

        lines.append("")
        lines.append("REFERENCES:")
        for i in range(1, NUM_SLOTS + 1):
            lines.append(f"Slot {i}: {'OK' if refs_saved[i] else 'MISSING'}")

        lines.append("")
        lines.append("CHECKS:")
        for i in range(1, NUM_SLOTS + 1):
            check = checks_copy[i]
            if check is None:
                lines.append(f"Slot {i}: MISSING")
            else:
                lines.append(
                    f"Slot {i}: {check['status']} | "
                    f"N={check['score_normal']:.3f} | "
                    f"R={check['score_rot180']:.3f} | "
                    f"image={check['image']}"
                )

        lines.append("")
        lines.append("FILES:")
        for key, path in paths.items():
            lines.append(f"{key}: {path}")

        lines.append("")
        lines.append("LAST EVENTS:")
        if events_tail:
            for ev in events_tail:
                lines.append(ev)
        else:
            lines.append("No events recorded.")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        paths["text"] = txt_path

        with self.lock:
            self.last_report_path = txt_path

        self.add_event(f"Debug snapshot saved: {txt_path}")

        return True, f"Debug snapshot saved: {txt_path}"

    # ========================================================
    # REPORT
    # ========================================================

    def save_report(self):
        ensure_dirs()

        with self.lock:
            status_text = self.status_text
            result_text = self.result_text
            workflow_active = self.workflow_active
            motion_active = self.motion_active
            cmd_vel_subscribers = self.cmd_vel_subscriber_count()

            refs_copy = dict(self.references)
            ref_times_copy = dict(self.reference_times)

            checks_copy = {
                i: self.checks[i].copy() if self.checks[i] is not None else None
                for i in range(1, NUM_SLOTS + 1)
            }

            events_copy = list(self.event_log)

        report_path = os.path.join(
            LOG_DIR,
            f"five_slot_robot_assisted_report_{now_file_str()}.txt"
        )

        missing_refs = [
            i for i in range(1, NUM_SLOTS + 1)
            if refs_copy[i] is None
        ]

        missing_checks = [
            i for i in range(1, NUM_SLOTS + 1)
            if checks_copy[i] is None
        ]

        rotated_slots = [
            i for i in range(1, NUM_SLOTS + 1)
            if checks_copy[i] is not None and checks_copy[i]["status"] == "ROTATED_180"
        ]

        lines = []
        lines.append("========================================")
        lines.append("Five Slot Robot Assisted Debug Report")
        lines.append("========================================")
        lines.append(f"Created: {now_str()}")
        lines.append(f"Last status: {status_text}")
        lines.append(f"Workflow active: {workflow_active}")
        lines.append(f"Motion active: {motion_active}")
        lines.append(f"Cmd_vel subscribers: {cmd_vel_subscribers}")
        lines.append(f"Base health: {self.base_health_text()}")
        lines.append("")
        lines.append("CONFIG:")
        lines.append(f"PORT = {PORT}")
        lines.append(f"NUM_SLOTS = {NUM_SLOTS}")
        lines.append(f"CAMERA_RAW_TOPIC = {CAMERA_RAW_TOPIC}")
        lines.append(f"CAMERA_COMPRESSED_TOPIC = {CAMERA_COMPRESSED_TOPIC}")
        lines.append(f"CMD_VEL_TOPIC = {CMD_VEL_TOPIC}")
        lines.append(f"TURN_SPEED = {TURN_SPEED}")
        lines.append(f"TURN_SMALL = {TURN_SMALL}")
        lines.append(f"TURN_MEDIUM = {TURN_MEDIUM}")
        lines.append(f"TURN_NEXT_DURATION = {TURN_NEXT_DURATION}")
        lines.append(f"ARC_LINEAR_SPEED = {ARC_LINEAR_SPEED}")
        lines.append(f"ARC_TURN_SPEED = {ARC_TURN_SPEED}")
        lines.append(f"ARC_NEXT_DURATION = {ARC_NEXT_DURATION}")
        lines.append(f"SLOT_TURN_SPEED = {SLOT_TURN_SPEED}")
        lines.append(f"SLOT_TURN_90_DURATION = {SLOT_TURN_90_DURATION}")
        lines.append(f"SLOT_TURN_180_DURATION = {SLOT_TURN_180_DURATION}")
        lines.append(f"SLOT_FORWARD_SPEED = {SLOT_FORWARD_SPEED}")
        lines.append(f"SLOT_FORWARD_DURATION = {SLOT_FORWARD_DURATION}")
        lines.append(f"DEMO_WAIT_SECONDS = {DEMO_WAIT_SECONDS}")
        lines.append(f"AUTO_CENTER_BEFORE_CAPTURE = {AUTO_CENTER_BEFORE_CAPTURE}")
        lines.append(f"AUTO_CENTER_MAX_STEPS = {AUTO_CENTER_MAX_STEPS}")
        lines.append(f"AUTO_CENTER_TOLERANCE_RATIO = {AUTO_CENTER_TOLERANCE_RATIO}")
        lines.append(f"LOOK_AWAY_RIGHT_DURATION = {LOOK_AWAY_RIGHT_DURATION}")
        lines.append(f"LOOK_AWAY_LEFT_DURATION = {LOOK_AWAY_LEFT_DURATION}")
        lines.append(f"MIN_CARD_AREA = {MIN_CARD_AREA}")
        lines.append(f"MIN_GOOD_SCORE = {MIN_GOOD_SCORE}")
        lines.append(f"ROTATION_MARGIN = {ROTATION_MARGIN}")
        lines.append(f"CARD_BRIGHTNESS_MIN = {CARD_BRIGHTNESS_MIN}")
        lines.append(f"CARD_BRIGHTNESS_OTSU_MARGIN = {CARD_BRIGHTNESS_OTSU_MARGIN}")
        lines.append(f"CARD_SATURATION_MAX = {CARD_SATURATION_MAX}")
        lines.append(f"CARD_REGION_TOP_FRAC = {CARD_REGION_TOP_FRAC}")
        lines.append(f"CARD_REGION_BOTTOM_FRAC = {CARD_REGION_BOTTOM_FRAC}")
        lines.append(f"CARD_MIN_ASPECT = {CARD_MIN_ASPECT}")
        lines.append(f"CARD_MAX_ASPECT = {CARD_MAX_ASPECT}")
        lines.append(f"CARD_PEDESTAL_PENALTY = {CARD_PEDESTAL_PENALTY}")
        lines.append(f"TWO_STEP_EDGE_REJECT_FRAC = {TWO_STEP_EDGE_REJECT_FRAC}")
        lines.append(f"TWO_STEP_CHECK_FLEXIBLE_MATCH = {TWO_STEP_CHECK_FLEXIBLE_MATCH}")
        lines.append(f"TWO_STEP_MIN_FLEX_MATCH_SCORE = {TWO_STEP_MIN_FLEX_MATCH_SCORE}")
        lines.append(f"TWO_STEP_COMBINED_MATCH_WEIGHT = {TWO_STEP_COMBINED_MATCH_WEIGHT}")
        lines.append(f"TWO_STEP_COMBINED_QUALITY_WEIGHT = {TWO_STEP_COMBINED_QUALITY_WEIGHT}")
        lines.append(f"TWO_STEP_QUALITY_REPLACE_MARGIN = {TWO_STEP_QUALITY_REPLACE_MARGIN}")
        lines.append(f"ONE_SHOT_REQUIRED_CARDS = {ONE_SHOT_REQUIRED_CARDS}")
        lines.append(f"ONE_SHOT_MAX_CANDIDATES = {ONE_SHOT_MAX_CANDIDATES}")
        lines.append(f"ONE_SHOT_MIN_CARD_QUALITY = {ONE_SHOT_MIN_CARD_QUALITY}")
        lines.append(f"ONE_SHOT_EDGE_REJECT_FRAC = {ONE_SHOT_EDGE_REJECT_FRAC}")
        lines.append(f"CAMERA_FROZEN_TIMEOUT = {CAMERA_FROZEN_TIMEOUT}")
        lines.append("")
        lines.append("REFERENCES:")
        for i in range(1, NUM_SLOTS + 1):
            if refs_copy[i] is None:
                lines.append(f"Slot {i}: MISSING")
            else:
                lines.append(f"Slot {i}: SAVED at {ref_times_copy[i]}")

        lines.append("")
        lines.append("CHECKS:")
        for i in range(1, NUM_SLOTS + 1):
            c = checks_copy[i]
            if c is None:
                lines.append(f"Slot {i}: MISSING_CHECK")
            else:
                extra = ""
                if c.get("flex_source"):
                    extra += f" | source={c.get('flex_source')} det={c.get('flex_detection_index')}"
                if c.get("quality") is not None:
                    extra += f" | quality={float(c.get('quality')):.3f}"
                if c.get("combined_score") is not None:
                    extra += f" | combined={float(c.get('combined_score')):.3f}"
                lines.append(
                    f"Slot {i}: {c['status']} | "
                    f"score_normal={c['score_normal']:.3f} | "
                    f"score_rot180={c['score_rot180']:.3f} | "
                    f"time={c['time']} | "
                    f"image={c['image']}" + extra
                )

        lines.append("")
        lines.append("FINAL RESULT:")
        lines.append(result_text)
        lines.append(f"Rotated slots: {rotated_slots}")
        lines.append(f"Missing references: {missing_refs}")
        lines.append(f"Missing checks: {missing_checks}")

        with self.lock:
            cm = self.cross_match_result

        if cm is not None:
            lines.append("")
            lines.append("CROSS-MATCH ANALYSIS:")
            lines.append(f"Summary: {cm['summary']}")
            if cm.get("rotated_refs"):
                lines.append(f"Rotated reference slots: {cm['rotated_refs']}")
            if cm.get("moved"):
                lines.append(f"Slot order shifts: {cm['moved']}")
            if cm.get("low_conf"):
                lines.append(f"Low-confidence check slots: {cm['low_conf']}")
            lines.append("")
            lines.append("PER-CHECK BEST MATCH:")
            for cs in range(1, NUM_SLOTS + 1):
                info = cm["per_check"].get(cs, {})
                if info.get("status") != "MATCHED":
                    lines.append(f"Check Slot {cs}: {info.get('status', 'N/A')}")
                    continue
                lines.append(
                    f"Check Slot {cs} -> Reference Slot {info['best_ref_slot']} "
                    f"({info['best_orientation']}, score={info['best_score']:.3f})"
                )
                # Detail: scores against each reference
                all_s = info.get("all_scores", {})
                for rs in sorted(all_s.keys()):
                    s = all_s[rs]
                    lines.append(
                        f"    vs Ref {rs}: normal={s['normal']:.3f} | "
                        f"rot180={s['rot180']:.3f}"
                    )

        lines.append("")
        lines.append("EVENT LOG:")
        if not events_copy:
            lines.append("No events recorded.")
        else:
            for ev in events_copy:
                lines.append(ev)

        lines.append("")
        lines.append("========================================")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        with self.lock:
            self.last_report_path = report_path

        self.add_event(f"Report saved: {report_path}")

        return report_path
