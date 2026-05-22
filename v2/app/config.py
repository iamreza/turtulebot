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


PORT = 8086

CAMERA_RAW_TOPIC = "/camera/image_raw"
CMD_VEL_TOPIC = "/cmd_vel"
ODOM_TOPIC = "/odom"
BATTERY_TOPIC = "/battery_state"
NUM_SLOTS = 5

# Robot motion (odom closed-loop turns). See app/camera_node.py turn_by_angle_blocking.
# Staged bring-up: keep this False to test scan -> wait -> rescan -> rotation
# detection with a MANUAL card flip (no robot motion). Flip to True once the
# vision pipeline is verified, to re-enable the 180 deg turns (stage 4).
ROBOT_TURN_ENABLED = True
SLOT_TURN_SPEED = 0.40       # rad/s
CMD_PUBLISH_PERIOD = 0.15    # s between /cmd_vel publishes during a turn
BASE_ALIVE_TIMEOUT = 2.0     # no /odom for this long -> base considered dead
DEMO_WAIT_SECONDS = 15.0     # human-flip wait between the two scans

# Camera settle + sharpest-frame capture. After a 180 deg turn the TurtleBot3
# chassis keeps drifting briefly, so a frame grabbed immediately is motion
# blurred and breaks orientation matching. Before each scan we let the chassis
# settle, then sample several distinct frames and keep the sharpest (highest
# variance-of-Laplacian focus measure).
SCAN_SETTLE_SECONDS = 1.5       # wait before the CHECK scan (after the turn-back)
BASELINE_SETTLE_SECONDS = 0.3   # baseline has no preceding motion -> short settle
SHARPEST_FRAME_SECONDS = 1.0    # sample frames for this long and keep the sharpest

# Orientation comparison (reference scan vs check scan). Mirrors v1.
MIN_GOOD_SCORE = 0.35        # below this, verdict is LOW_CONFIDENCE
ROTATION_MARGIN = 0.05       # rot180 must beat normal by this to flag ROTATED_180

# راهکار 1 — center-fine orientation override. Low-detail / near-symmetric cards
# (e.g. an Ace) carry their only orientation cue in the central pip. The global
# multiscale match downscales + blurs + equalizes that cue away and can report a
# rotated card as NORMAL with false confidence. We re-compare ONLY the central
# region at higher resolution with detail-preserving prep (CLAHE, no blur) and,
# when it clearly disagrees with the global verdict, let it override.
FINE_W = 480                 # detail-preserving canonical size (2x the pipeline crop)
FINE_H = 720
FINE_CENTER_MARGINS = (0.26, 0.30, 0.34, 0.38, 0.42)  # central bands; pick most confident
FINE_MARGIN = 0.05           # min |r_fine - n_fine| to trust the center signal

# Playing-card detector tuning. Cards are expected to be white-ish, vertical
# rectangles on a dark background, arranged in one horizontal row.
CARD_MIN_AREA_FRAC = 0.003
CARD_MAX_AREA_FRAC = 0.35
CARD_MIN_ASPECT = 0.35
CARD_MAX_ASPECT = 0.95
CARD_MIN_SOLIDITY = 0.20
CARD_CLOSE_KERNEL = 15
CARD_BRIGHTNESS_FLOOR = 135
CARD_SATURATION_MAX = 125
CARD_OTSU_MARGIN = 12
CARD_ROW_RUN_REJECT_FRAC = 0.55
CARD_ROW_Y_TOL_FRAC = 0.22
CARD_NMS_IOU = 0.30

CROP_W = 240
CROP_H = 360

LOG_DIR = "logs"
SESSION_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(LOG_DIR, f"v2_{SESSION_STAMP}")
LIVE_LOG_FILE = os.path.join(LOG_DIR, "v2_live.log")

LATEST_RAW = os.path.join(RUN_DIR, "raw_frame.png")
LATEST_MASK = os.path.join(RUN_DIR, "mask.png")
LATEST_OVERLAY = os.path.join(RUN_DIR, "contour_overlay.png")
