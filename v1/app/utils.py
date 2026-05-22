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
import math
import os
import sys
import threading

import cv2
import numpy as np

from .config import *

# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)


class TeeStream:
    """Writes to original stream AND a log file. Used for stdout/stderr."""

    def __init__(self, original, file_path, tag):
        self.original = original
        self.file_path = file_path
        self.tag = tag
        self.lock = threading.Lock()

    def write(self, data):
        try:
            self.original.write(data)
            self.original.flush()
        except Exception:
            pass

        try:
            text = data if data.endswith("\n") or data == "" else data
            with self.lock:
                with open(self.file_path, "a", encoding="utf-8", errors="replace") as f:
                    if text and text.strip():
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        for line in text.rstrip("\n").split("\n"):
                            f.write(f"{ts} | {self.tag} | {line}\n")
                    else:
                        f.write(text)
        except Exception:
            pass

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self.original.isatty()
        except Exception:
            return False

    def fileno(self):
        return self.original.fileno()


def setup_session_logging():
    """Redirect sys.stdout and sys.stderr through TeeStream so everything
    printed goes both to the terminal and to the session log file.
    Also creates latest_session.log / latest_trace.log convenience symlinks."""
    ensure_dirs()

    sys.stdout = TeeStream(sys.__stdout__, SESSION_LOG_FILE, "STDOUT")
    sys.stderr = TeeStream(sys.__stderr__, SESSION_LOG_FILE, "STDERR")

    for link, target_basename in (
        (LATEST_SESSION_LINK, os.path.basename(SESSION_LOG_FILE)),
        (LATEST_TRACE_LINK, os.path.basename(TRACE_LOG_FILE)),
    ):
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(target_basename, link)
        except Exception as e:
            print(f"[SESSION LOG] Could not create symlink {link}: {e}", flush=True)

    print("=" * 60, flush=True)
    print(f"[SESSION LOG] stdout/stderr -> {SESSION_LOG_FILE}", flush=True)
    print(f"[SESSION LOG] trace events  -> {TRACE_LOG_FILE}", flush=True)
    print(f"[SESSION LOG] event log     -> {LIVE_LOG_FILE}", flush=True)
    print(f"[SESSION LOG] latest links  -> {LATEST_SESSION_LINK}, {LATEST_TRACE_LINK}", flush=True)
    print("=" * 60, flush=True)


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def quaternion_to_yaw(q):
    # Z-axis rotation (yaw) extracted from a geometry_msgs Quaternion.
    # Returns radians in (-pi, pi]; positive is counter-clockwise from +X.
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def shortest_angle_diff(a, b):
    # Smallest signed difference a - b, normalized to (-pi, pi]
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d <= -math.pi:
        d += 2.0 * math.pi
    return d


def normalize_angle(a):
    # Normalize angle to (-pi, pi]
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def now_file_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def put_lines(img, lines, x=10, y=30, dy=30, scale=0.75, color=(255, 255, 255)):
    yy = y

    for line in lines:
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA
        )
        yy += dy


def resize_same_height(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = max(h1, h2)

    if h1 != target_h:
        new_w1 = int(w1 * target_h / h1)
        img1 = cv2.resize(img1, (new_w1, target_h))

    if h2 != target_h:
        new_w2 = int(w2 * target_h / h2)
        img2 = cv2.resize(img2, (new_w2, target_h))

    return img1, img2


def prepare_crop(crop_bgr):
    # Normalize the input crop by finding the actual white-card boundary
    # inside it and resizing only the card to a canonical 220x320 grayscale.
    # This removes scan-to-scan variability caused by different bbox sizes
    # (some scans include more wood pedestal, some less) — without this
    # normalization, the same card photographed twice can produce prepared
    # images that match poorly because the card is at different scale/position.
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]

    if h < 20 or w < 20:
        return None

    refined = _refine_to_card_only(crop_bgr)
    if refined is None or refined.size == 0:
        # fall back to small margin trim if refinement failed
        mx = max(2, int(w * 0.03))
        my = max(2, int(h * 0.03))
        refined = crop_bgr[my:h - my, mx:w - mx]
        if refined is None or refined.size == 0:
            return None

    gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    return gray


def _refine_to_card_only(crop_bgr):
    # Inside `crop_bgr` (the detector's bbox crop, which may include wood
    # pedestal or background), find the actual playing-card boundary.
    #
    # Strategy uses BOTH brightness (V) and saturation (S):
    #   - White card: V > 180 AND S < ~60 (white = no color)
    #   - Wood pedestal: V ~ 150-200 BUT S > 60 (yellow/brown = colored)
    #   - Background: V < 100
    # This rejects the wood even when it's bright enough on V alone.
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Cards are nearly pure white (S~3-15). Wood pedestals are yellow-ish
    # (S~50-90). A strict low-saturation threshold separates them cleanly.
    bright = cv2.inRange(v, 170, 255)
    low_sat = cv2.inRange(s, 0, 35)
    mask = cv2.bitwise_and(bright, low_sat)

    # Modest close to fill in card interior (text/pips/artwork). The colored
    # picture inside the card has S high (so it's not in the mask), so we
    # must close to fill those holes — but not so much we bridge to wood.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest portrait-ish contour whose top is in the upper half
    # (so wood-only contours, even if they pass the saturation test for some
    # reason, are excluded).
    cx_frame = w / 2.0
    candidates = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        bbox_area = cw * ch
        if bbox_area < (w * h * 0.08):
            continue
        aspect = cw / float(max(ch, 1))
        if aspect < 0.30 or aspect > 1.10:  # cards are portrait
            continue
        if cy > h * 0.55:  # contour starts below mid-frame -> likely wood
            continue
        center_x = cx + cw / 2.0
        x_offset = abs(center_x - cx_frame) / float(w)
        candidates.append({
            "bbox": (cx, cy, cw, ch),
            "bbox_area": bbox_area,
            "x_offset": x_offset,
        })

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda c: c["bbox_area"] * (1.0 - 0.5 * c["x_offset"])
    )
    cx, cy, cw, ch = best["bbox"]

    area_ratio = (cw * ch) / float(w * h)
    if area_ratio < 0.10:
        return None

    pad = 3
    cx = max(0, cx - pad)
    cy = max(0, cy - pad)
    cw = min(w - cx, cw + 2 * pad)
    ch = min(h - cy, ch + 2 * pad)

    return crop_bgr[cy:cy + ch, cx:cx + cw]


def compare_score(img_a, img_b):
    if img_a is None or img_b is None:
        return -1.0

    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)

    score = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0][0]
    return float(score)


def prepare_orientation_area(prepared_gray, margin=0.08):
    # Crop a symmetric inner region at the given margin fraction.  Used by
    # the orientation comparison; calling at multiple margins lets us focus
    # either on the corners (small margin) or on the center of the card
    # (large margin), which is critical for rotationally-symmetric borders
    # like traffic-sign cards where only the central content rotates.
    if prepared_gray is None or prepared_gray.size == 0:
        return None

    h, w = prepared_gray.shape[:2]
    mx = max(2, int(w * margin))
    my = max(2, int(h * margin))

    inner = prepared_gray[my:h - my, mx:w - mx]

    if inner is None or inner.size == 0:
        return None

    return cv2.resize(inner, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


# Margin levels for multi-scale orientation comparison.  We probe at narrow
# (preserves corner indices on playing cards), medium, and aggressive (focuses
# on central content for cards with rotation-invariant borders).
_ORIENTATION_MARGINS = (0.08, 0.20, 0.32)


def compare_orientation_score(img_a, img_b, margin=0.08):
    # Single-margin comparison at the given margin (default keeps the original
    # 8% behaviour for backward compatibility).
    a = prepare_orientation_area(img_a, margin)
    b = prepare_orientation_area(img_b, margin)
    return compare_score(a, b)


def compare_orientation_pair_multiscale(ref_prepared, check_prepared):
    # Returns (normal_score, rot180_score) — the pair of (normal, rot180)
    # comparison scores at the crop margin that gives the strongest
    # rotation-vs-normal signal (i.e., maximum rot180 - normal).
    #
    # Why: card designs vary.  Standard playing cards put the rotation cue
    # in the corner indices (small margin works best).  Traffic-sign cards
    # have rotationally-symmetric borders and put the cue only in the
    # central design (large margin works best).  Rather than pick one, we
    # try several and use the margin where rotation evidence is clearest.
    if ref_prepared is None or check_prepared is None:
        return (-1.0, -1.0)

    ref_rot = cv2.rotate(ref_prepared, cv2.ROTATE_180)
    best_n, best_r, best_delta = -1.0, -1.0, -2.0

    for m in _ORIENTATION_MARGINS:
        a = prepare_orientation_area(ref_prepared, m)
        b = prepare_orientation_area(check_prepared, m)
        ar = prepare_orientation_area(ref_rot, m)
        if a is None or b is None or ar is None:
            continue
        n = compare_score(a, b)
        r = compare_score(ar, b)
        delta = r - n
        if delta > best_delta:
            best_delta = delta
            best_n, best_r = n, r

    return float(best_n), float(best_r)

