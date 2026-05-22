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

import cv2
import numpy as np

from .config import (
    CROP_W,
    CROP_H,
    FINE_W,
    FINE_H,
    FINE_CENTER_MARGINS,
    FINE_MARGIN,
)


# ============================================================
# Geometry helpers (odom yaw)
# ============================================================

def quaternion_to_yaw(q):
    # Z-axis rotation (yaw) from a geometry_msgs Quaternion. Radians in (-pi, pi].
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
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


# ============================================================
# Crop normalization + orientation comparison (ported from v1)
# ============================================================

def prepare_crop(crop_bgr):
    # Normalize the input crop by finding the actual white-card boundary inside
    # it and resizing only the card to a canonical grayscale image. This removes
    # scan-to-scan variability from differing bbox sizes so the same card scanned
    # twice produces comparable prepared images.
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    refined = _refine_to_card_only(crop_bgr)
    if refined is None or refined.size == 0:
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
    # Inside the detector's bbox crop (which may include pedestal/background),
    # find the actual playing-card boundary using brightness (V) + saturation (S):
    #   white card: V high AND S low; wood/colored: S high; background: V low.
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    bright = cv2.inRange(v, 170, 255)
    low_sat = cv2.inRange(s, 0, 35)
    mask = cv2.bitwise_and(bright, low_sat)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

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
        if cy > h * 0.55:  # contour starts below mid-frame -> likely pedestal
            continue
        center_x = cx + cw / 2.0
        x_offset = abs(center_x - cx_frame) / float(w)
        candidates.append({"bbox": (cx, cy, cw, ch), "bbox_area": bbox_area, "x_offset": x_offset})

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c["bbox_area"] * (1.0 - 0.5 * c["x_offset"]))
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

    h, w = img_a.shape[:2]
    if h < 3 or w < 3:
        return -1.0

    dy = max(1, int(h * 0.05))
    dx = max(1, int(w * 0.05))
    template = img_a[dy:h - dy, dx:w - dx]
    if template is None or template.size == 0:
        return -1.0

    res = cv2.matchTemplate(
        img_b.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)


def prepare_orientation_area(prepared_gray, margin=0.08):
    # Crop a symmetric inner region at the given margin fraction. Small margins
    # preserve corner indices; large margins focus on central content.
    if prepared_gray is None or prepared_gray.size == 0:
        return None
    h, w = prepared_gray.shape[:2]
    mx = max(2, int(w * margin))
    my = max(2, int(h * margin))
    inner = prepared_gray[my:h - my, mx:w - mx]
    if inner is None or inner.size == 0:
        return None
    return cv2.resize(inner, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


# Margins for multi-scale orientation comparison: narrow (corner indices),
# medium, aggressive (central content for rotation-invariant borders).
_ORIENTATION_MARGINS = (0.08, 0.20, 0.32)


def compare_orientation_score(img_a, img_b, margin=0.08):
    a = prepare_orientation_area(img_a, margin)
    b = prepare_orientation_area(img_b, margin)
    return compare_score(a, b)


def compare_orientation_pair_multiscale(ref_prepared, check_prepared):
    # Returns (normal_score, rot180_score) at the crop margin where the match is
    # most confident, i.e. the margin maximizing max(normal, rot180). Both scores
    # are reported from that single margin so the verdict compares them at the
    # same scale.
    #
    # Earlier this picked the margin maximizing the rotation signal (rot180 -
    # normal). That backfired: an aggressive margin can maximize the *gap* while
    # collapsing both absolute scores below MIN_GOOD_SCORE, so a genuinely
    # rotated card (rot180 clearly > normal at every margin) was reported with a
    # tiny absolute rot180 and graded LOW_CONFIDENCE instead of ROTATED_180.
    if ref_prepared is None or check_prepared is None:
        return (-1.0, -1.0)

    ref_rot = cv2.rotate(ref_prepared, cv2.ROTATE_180)
    best_n, best_r, best_conf = -1.0, -1.0, -2.0

    for m in _ORIENTATION_MARGINS:
        a = prepare_orientation_area(ref_prepared, m)
        b = prepare_orientation_area(check_prepared, m)
        ar = prepare_orientation_area(ref_rot, m)
        if a is None or b is None or ar is None:
            continue
        n = compare_score(a, b)
        r = compare_score(ar, b)
        conf = max(n, r)
        if conf > best_conf:
            best_conf = conf
            best_n, best_r = n, r

    return float(best_n), float(best_r)


# ============================================================
# راهکار 1 — center-fine orientation (floating central pip match)
# ============================================================

def prepare_crop_fine(crop_bgr):
    # Detail-preserving counterpart to prepare_crop, for the orientation cue that
    # lives in the central pip of low-detail cards. Higher canonical resolution,
    # local CLAHE instead of global equalizeHist, and NO Gaussian blur — so a
    # tiny asymmetric pip survives instead of being averaged into the white card.
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    refined = _refine_to_card_only(crop_bgr)
    if refined is None or refined.size == 0:
        mx = max(2, int(w * 0.03))
        my = max(2, int(h * 0.03))
        refined = crop_bgr[my:h - my, mx:w - mx]
        if refined is None or refined.size == 0:
            return None

    gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (FINE_W, FINE_H), interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def _center_region_raw(prepared, margin):
    # Return the raw central crop. Resizing this tiny high-detail region would
    # amplify detector jitter and make the pip comparison translation-sensitive.
    if prepared is None or prepared.size == 0:
        return None
    h, w = prepared.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    inner = prepared[my:h - my, mx:w - mx]
    if inner is None or inner.size == 0:
        return None
    return inner


def compare_score_sliding(template, search_img):
    # Let the smaller reference template float inside the larger check image, so
    # tiny bbox/refinement shifts do not dominate the fine orientation score.
    if template is None or search_img is None:
        return -1.0

    th, tw = template.shape[:2]
    sh, sw = search_img.shape[:2]
    if th <= 0 or tw <= 0 or sh <= 0 or sw <= 0:
        return -1.0

    if th > sh or tw > sw:
        template = cv2.resize(template, (sw, sh), interpolation=cv2.INTER_AREA)

    res = cv2.matchTemplate(
        search_img.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)


def compare_center_fine_pair(ref_fine, check_fine):
    # Compare ONLY the central region in normal vs rot180, over a band of central
    # margins, returning the (normal, rot180) scores at the margin with the most
    # confident rotation signal (max r - n). Isolating the pip is what recovers
    # the orientation of a near-symmetric card the global match cannot resolve.
    if ref_fine is None or check_fine is None:
        return (-1.0, -1.0)

    ref_rot = cv2.rotate(ref_fine, cv2.ROTATE_180)
    best_n, best_r, best_gap = -1.0, -1.0, -4.0
    for m in FINE_CENTER_MARGINS:
        search_margin = max(0.0, m - 0.05)
        b = _center_region_raw(check_fine, search_margin)
        a = _center_region_raw(ref_fine, m)
        ar = _center_region_raw(ref_rot, m)
        if a is None or b is None or ar is None:
            continue
        n = compare_score_sliding(a, b)
        r = compare_score_sliding(ar, b)
        gap = r - n
        if gap > best_gap:
            best_gap = gap
            best_n, best_r = n, r
    return float(best_n), float(best_r)


def orientation_verdict(score_normal, score_rot180, min_good, margin,
                        fine_normal=None, fine_rot180=None):
    # Global verdict first; it is reliable for detailed, clearly asymmetric cards.
    if max(score_normal, score_rot180) < min_good:
        base = "LOW_CONFIDENCE"
    elif score_rot180 > score_normal + margin:
        base = "ROTATED_180"
    elif score_normal > score_rot180 + margin:
        base = "NORMAL"
    else:
        base = "UNCLEAR"

    # راهکار 1 override: when the global verdict is not a confident ROTATED_180
    # but the high-res central pip clearly indicates rotation, trust the pip.
    # Normal cards show a strongly negative center gap, so a positive gap beyond
    # FINE_MARGIN is a reliable rotation signal even when its absolute scores are
    # small (the pip is a tiny fraction of the card).
    if fine_normal is None or fine_rot180 is None:
        return base
    fine_gap = fine_rot180 - fine_normal
    if base != "ROTATED_180" and fine_gap > FINE_MARGIN:
        return "ROTATED_180"
    if base == "ROTATED_180" and fine_gap < -FINE_MARGIN:
        return "NORMAL"
    return base
