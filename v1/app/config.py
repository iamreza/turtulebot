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

import datetime
import os

# ============================================================
# CONFIG
# ============================================================

PORT = 8086

CAMERA_RAW_TOPIC = "/camera/image_raw"
CAMERA_COMPRESSED_TOPIC = "/camera/image_raw/compressed"
CMD_VEL_TOPIC = "/cmd_vel"
BATTERY_TOPIC = "/battery_state"

NUM_SLOTS = 5

# Two-step wide scan mode:
# View A should see the left side of the row (normally cards 1-3).
# View B should see the right side of the row (normally cards 3-5, with card 3 as overlap).
TWO_STEP_VIEW_A_SLOTS = [1, 2, 3]
TWO_STEP_VIEW_B_SLOTS = [3, 4, 5]
TWO_STEP_MAX_CARDS_PER_VIEW = 5
# Extra robustness for future automatic movement:
# - reject objects touching frame borders (walls/bright edges)
# - during CHECK, do not trust View A/B slot order; match every visible crop
#   against all references and store it under the best reference slot.
TWO_STEP_EDGE_REJECT_FRAC = 0.040
TWO_STEP_CHECK_FLEXIBLE_MATCH = True
TWO_STEP_MIN_FLEX_MATCH_SCORE = 0.18
# v3: when the same card appears in both views, prefer the crop that is not
# only a good match, but also complete, centered and not close to the image edge.
TWO_STEP_QUALITY_REPLACE_MARGIN = 0.015
TWO_STEP_COMBINED_MATCH_WEIGHT = 0.78
TWO_STEP_COMBINED_QUALITY_WEIGHT = 0.22
TWO_STEP_NMS_IOU_THRESHOLD = 0.25
TWO_STEP_MIN_BASELINE_A_CARDS = 3
TWO_STEP_MIN_BASELINE_B_CARDS = 2
TWO_STEP_MIN_CHECK_TOTAL_COVERAGE = 5

# One-shot full-row scan mode:
# The robot sees all five cards in a single wide frame. Baseline assigns
# slots left-to-right; check stays flexible and matches every crop against all
# references, so small robot pose errors do not break the result.
ONE_SHOT_REQUIRED_CARDS = 5
ONE_SHOT_MAX_CANDIDATES = 12
# One-shot uses the whole wide image, so side cards may legitimately be close
# to the image edge. Keep two-step strict, but make one-shot permissive and
# let the 5-card row picker choose the real row.
ONE_SHOT_MIN_CARD_QUALITY = 0.26
ONE_SHOT_EDGE_REJECT_FRAC = 0.006
ONE_SHOT_ROW_Y_WEIGHT = 0.30
ONE_SHOT_QUALITY_WEIGHT = 0.70

# Full auto one-scan calibration. Used after the robot turns back toward the cards.
# Goal: get one stable frame where all five cards are visible with safe margins.
AUTO_ONESCAN_CALIBRATION_ENABLED = True
AUTO_ONESCAN_CALIBRATION_MAX_STEPS = 8
# v7: never back up endlessly during calibration. If five cards are not visible,
# first sweep/restore yaw; only then use a very small bounded distance correction.
AUTO_ONESCAN_MAX_BACK_TOTAL_M = 0.10
AUTO_ONESCAN_MAX_FORWARD_TOTAL_M = 0.06
AUTO_ONESCAN_SWEEP_DEGREES = [0, -6, 6, -12, 12, -18, 18, -24, 24, -30, 30]
# v8: yaw is a coarse helper, not a hard gate. The camera/vision result decides.
AUTO_ONESCAN_YAW_CORRECTION_SPEED = 0.12
AUTO_ONESCAN_RETURN_YAW_TOL_DEG = 8.0
AUTO_ONESCAN_RETURN_YAW_MAX_RESIDUAL_DEG = 28.0
AUTO_ONESCAN_AWAY_ACCEPT_RESIDUAL_DEG = 25.0
AUTO_ONESCAN_SWEEP_MIN_IMPROVEMENT = 1
AUTO_ONESCAN_CENTER_TOL_RATIO = 0.08
AUTO_ONESCAN_MIN_GROUP_WIDTH_RATIO = 0.55
AUTO_ONESCAN_MAX_GROUP_WIDTH_RATIO = 0.94
AUTO_ONESCAN_SAFE_MARGIN_RATIO = 0.025
AUTO_ONESCAN_TURN_SPEED = 0.14
AUTO_ONESCAN_TURN_DURATION_MIN = 0.08
AUTO_ONESCAN_TURN_DURATION_MAX = 0.35
AUTO_ONESCAN_MOVE_SPEED = 0.04
AUTO_ONESCAN_FORWARD_STEP_M = 0.03
AUTO_ONESCAN_BACKWARD_STEP_M = 0.04


# normal turning
TURN_SPEED = 0.12
TURN_SMALL = 0.2
TURN_MEDIUM = 0.5
TURN_NEXT_DURATION = 1.5

# arc movement: forward + turn
ARC_LINEAR_SPEED = 0.025
ARC_TURN_SPEED = 0.10
ARC_NEXT_DURATION = 1.2

# slot-to-slot movement: brisk turn-forward-turn pattern
# 90° turn = SLOT_TURN_SPEED * SLOT_TURN_90_DURATION rad (target: pi/2 = 1.571 rad)
# 180° turn = SLOT_TURN_SPEED * SLOT_TURN_180_DURATION rad (target: pi = 3.142 rad)
# Forward distance = SLOT_FORWARD_SPEED * SLOT_FORWARD_DURATION m (target: ~0.15 m)
SLOT_TURN_SPEED = 0.40        # rad/s (was 0.18)
SLOT_TURN_90_DURATION = 4.0   # 0.40 * 4.0 = 1.60 rad (~91.7° to allow for stiction)
SLOT_TURN_180_DURATION = 8.0  # 0.40 * 8.0 = 3.20 rad (~183.4°)
SLOT_FORWARD_SPEED = 0.08     # m/s (used by both timing and odom-distance moves)
SLOT_FORWARD_DURATION = 2.5   # legacy timing fallback
# Auto-run mode uses closed-loop distance based on odom position. We deliberately
# UNDERSHOOT the nominal slot spacing so the visual aligner only has to nudge
# forward (never backward). This is safer because backing up could collide with
# previous setup, and short overshoot is harder to recover from than undershoot.
SLOT_FORWARD_DISTANCE_M = 0.16  # 16 cm closed-loop, then vision aligns

# Visual alignment after each slot transition. Reads self.latest_bbox produced
# by the existing detect_best_card pipeline (NOT modified here). We only react
# to the detection result; we don't change how cards are detected.
VISION_ALIGN_ENABLED = True
VISION_ALIGN_MAX_STEPS = 8
VISION_ALIGN_TARGET_HEIGHT_RATIO = 0.55   # card height should be ~55% of frame
VISION_ALIGN_HEIGHT_TOL_RATIO = 0.10      # +/- 10% acceptable
VISION_ALIGN_X_TOL_RATIO = 0.06           # x-centering tolerance (same as before)
VISION_ALIGN_TURN_SPEED = 0.18
VISION_ALIGN_TURN_DURATION_MIN = 0.08
VISION_ALIGN_TURN_DURATION_MAX = 0.40
VISION_ALIGN_LINEAR_SPEED = 0.04
VISION_ALIGN_FORWARD_STEP_M = 0.04        # 4 cm per forward correction
VISION_ALIGN_BACKWARD_STEP_M = 0.03       # 3 cm per backward correction (smaller — safer)
VISION_ALIGN_SEARCH_FORWARD_M = 0.05      # 5 cm per search step
VISION_ALIGN_SEARCH_MAX_STEPS = 4         # up to 20 cm of forward search
# Hard cap on TOTAL forward distance moved during one vision_align call.
# Prevents the "robot drives into the card" failure mode where step landed
# the robot already-too-close, then forward search/correction kept advancing.
VISION_ALIGN_MAX_TOTAL_FORWARD_M = 0.10   # 10 cm cumulative forward limit
DEMO_WAIT_SECONDS = 15.0
CMD_PUBLISH_PERIOD = 0.15

LOOK_AWAY_RIGHT_DURATION = 1.8
LOOK_AWAY_LEFT_DURATION = 1.8
LOOK_BACK_LEFT_DURATION = LOOK_AWAY_RIGHT_DURATION
LOOK_BACK_RIGHT_DURATION = LOOK_AWAY_LEFT_DURATION

CARD_STABLE_SECONDS = 0.25
CARD_WAIT_TIMEOUT = 4.0
POST_MOVE_SETTLE_SECONDS = 0.45

ODOM_TOPIC = "/odom"
BASE_ALIVE_TIMEOUT = 2.0  # if no /odom for this long, base is considered dead

AUTO_CENTER_BEFORE_CAPTURE = False  # MANUAL MODE: user places robot, no robot motion
AUTO_CENTER_MAX_STEPS = 12       # was 8, more attempts to recover from drift
AUTO_CENTER_TOLERANCE_RATIO = 0.06
AUTO_CENTER_TURN_SPEED = 0.18    # was 0.12, faster correction per step
AUTO_CENTER_TURN_DURATION_MIN = 0.08   # smallest correction step (~ 0.8°)
AUTO_CENTER_TURN_DURATION_MAX = 0.40   # largest correction step (~ 4.1°)
# Step duration scales with error magnitude — big offsets get bigger turns,
# small offsets get small turns. Avoids both undershoot and overshoot.

MIN_CARD_AREA = 6000
MIN_GOOD_SCORE = 0.35
ROTATION_MARGIN = 0.05

# Detector tuning for WHITE CARDS on DARK BACKGROUND.
# Detects cards by brightness (V) + low saturation (S) in HSV space.
# Works regardless of background color (blue, purple, etc.).
CARD_BRIGHTNESS_MIN = 130    # absolute floor on V; Otsu picks threshold above this
CARD_BRIGHTNESS_OTSU_MARGIN = 5   # subtract from Otsu threshold to be slightly lenient
CARD_SATURATION_MAX = 130    # lenient; cream/warm white cards in warm light go up to ~120
CARD_REGION_TOP_FRAC = 0.03  # ignore top X% of frame
CARD_REGION_BOTTOM_FRAC = 0.97
CARD_MIN_ASPECT = 0.28       # bw/bh; cards are portrait
CARD_MAX_ASPECT = 1.10
CARD_MAX_WIDTH_FRAC = 0.70
CARD_MAX_HEIGHT_FRAC = 0.95
CARD_PEDESTAL_PENALTY = True # prefer card-shaped objects in upper part of frame

# Camera health
CAMERA_FROZEN_TIMEOUT = 5.0  # seconds without a frame -> CAMERA_FROZEN status

CROP_W = 220
CROP_H = 320

LOG_DIR = "logs"
LIVE_LOG_FILE = os.path.join(LOG_DIR, "five_slot_robot_assisted_live.log")
SESSION_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TRACE_LOG_FILE = os.path.join(LOG_DIR, f"trace_{SESSION_STAMP}.log")
SESSION_LOG_FILE = os.path.join(LOG_DIR, f"session_{SESSION_STAMP}.log")
LATEST_SESSION_LINK = os.path.join(LOG_DIR, "latest_session.log")
LATEST_TRACE_LINK = os.path.join(LOG_DIR, "latest_trace.log")
