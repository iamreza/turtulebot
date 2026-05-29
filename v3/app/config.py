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

# approach 1 — center-fine orientation override. Low-detail / near-symmetric cards
# (e.g. an Ace) carry their only orientation cue in the central pip. The global
# multiscale match downscales + blurs + equalizes that cue away and can report a
# rotated card as NORMAL with false confidence. We re-compare ONLY the central
# region at higher resolution with detail-preserving prep (CLAHE, no blur) and,
# when it clearly disagrees with the global verdict, let it override.
FINE_W = 480                 # detail-preserving canonical size (2x the pipeline crop)
FINE_H = 720
FINE_CENTER_MARGINS = (0.26, 0.30, 0.34, 0.38, 0.42)  # central bands; pick most confident
FINE_MARGIN = 0.15           # min |r_fine - n_fine| to trust the center signal.
                             # Raised from 0.05: a tiny central pip is noisy, and a
                             # small positive gap was enough to flip an unrotated
                             # card to a false ROTATED_180. A stricter band ignores
                             # that jitter and only acts on a decisive center signal.

# Playing-card detector tuning. Cards are white-ish rectangles on a dark
# background, arranged in one horizontal row.
#
# approach 3 (floor placement): the cards now lie FLAT on the floor and the fixed
# forward camera views them at an angle, so they image as foreshortened
# trapezoids. Geometry is measured with minAreaRect (see detector.py) and the
# aspect here is the short/long side RATIO (always <= 1): ~0.63 head-on, peaking
# toward 1.0 as foreshortening grows, so the band is widened accordingly. Area
# is also looser because near/far cards in the row differ more in size.
CARD_MIN_AREA_FRAC = 0.0025
CARD_MAX_AREA_FRAC = 0.35
CARD_MIN_ASPECT = 0.12  # extreme grazing angle squashes floor cards this flat/wide
CARD_MAX_ASPECT = 1.00
CARD_MIN_SOLIDITY = 0.40  # reject irregular backdrop glare blobs (clean cards are >0.7)
CARD_CLOSE_KERNEL = 5  # small kernel keeps adjacent floor cards from merging at grazing angle (was 15)
CARD_BRIGHTNESS_FLOOR = 135
CARD_SATURATION_MAX = 125
CARD_OTSU_MARGIN = 12
CARD_ROW_RUN_REJECT_FRAC = 0.55
CARD_ROW_Y_TOL_FRAC = 0.10  # floor cards share a near-identical cy (~10px spread);
                            # a tight band drops off-row side/upper junk cleanly
CARD_NMS_IOU = 0.30

# approach 3 (floor placement): detections hugging the top/left/right frame border
# are backdrop-wall glare / edge artifacts, never a full card lying on the floor
# (cards sit in the lower-central region). Reject them. The BOTTOM edge is NOT
# filtered — a near card can legitimately touch it.
CARD_EDGE_MARGIN_PX = 4
# Floor ROI: cards lie on the floor and image in the LOWER part of the frame, so
# any blob whose center sits in the upper band is backdrop-wall glare. Defense in
# depth alongside the top-edge filter — this also catches glare in the MIDDLE of
# the wall (which does not touch the top edge). Cards here center near 95% down,
# so a 0.50 line leaves a huge safety margin.
CARD_ROI_TOP_FRAC = 0.50
# Cap per-frame contour processing so a speckly backdrop/floor mask cannot spike
# CPU (which was hanging the Pi): keep only the largest-area contours.
CARD_MAX_CONTOURS = 40
# Expand the recovered card quad outward by this fraction before the rectify warp,
# to recapture the border eroded by the mask's MORPH_OPEN. Recovers the asymmetric
# detail that lets grazing-angle edge cards resolve NORMAL/ROTATED_180 instead of
# matching every orientation equally (UNCLEAR).
CARD_QUAD_EXPAND = 0.06
# Minimum quad-area / minAreaRect-area ratio for the rectify warp to be trusted.
# This is rotation-invariant (fill of the quad's own rotated bounding box), so a
# legitimately slanted grazing-angle edge card still passes (~0.8+); only a quad
# collapsed to a sliver falls below this and the detector then uses the padded
# axis-aligned fallback crop. NB: was previously measured against the axis-aligned
# bbox, which wrongly rejected slanted edge cards and left them un-rectified.
CARD_QUAD_MIN_FILL = 0.35
# After rectifying, a degenerate/mis-ordered quad warps to a near-uniform patch of
# off-card background (a blank crop that matches every orientation equally and
# grades UNCLEAR). Require the warped crop to carry real structure (grayscale std)
# before trusting it; below this the detector tries the rotated-rect rectify, then
# the axis-aligned fallback. std (not a white-pixel fraction) so it passes both
# white pip cards and colorful card backs.
CARD_CROP_MIN_STD = 12.0

CROP_W = 240
CROP_H = 360

LOG_DIR = "logs"
SESSION_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(LOG_DIR, f"v3_{SESSION_STAMP}")
LIVE_LOG_FILE = os.path.join(LOG_DIR, "v3_live.log")

LATEST_RAW = os.path.join(RUN_DIR, "raw_frame.png")
LATEST_MASK = os.path.join(RUN_DIR, "mask.png")
LATEST_OVERLAY = os.path.join(RUN_DIR, "contour_overlay.png")
