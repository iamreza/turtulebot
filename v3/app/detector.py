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

from dataclasses import dataclass

import cv2
import numpy as np

from .config import *


@dataclass
class CardCandidate:
    bbox: tuple
    area: float
    aspect: float
    solidity: float
    score: float
    reject_reason: str = ""
    quad: object = None  # 4 ordered corners (tl, tr, br, bl) for perspective rectify
    rect_quad: object = None  # min-area rotated-rect corners; robust rectify fallback


@dataclass
class DetectionResult:
    cards: list
    mask: np.ndarray
    overlay: np.ndarray
    crops: list
    rejected: list
    contour_count: int

    @property
    def ok(self):
        return len(self.cards) == NUM_SLOTS


def _bbox_iou(a, b):
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


def build_card_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Cards always sit on the floor in the lower half of the frame. Restrict the
    # Otsu computation to that band so a huge specular glare blob on the blue
    # backdrop (upper half) can't bias the threshold downward and let faint floor
    # reflections between cards leak into the mask, merging adjacent cards.
    roi_top = int(frame.shape[0] * CARD_ROI_TOP_FRAC)
    otsu_thr, _ = cv2.threshold(v[roi_top:, :], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(CARD_BRIGHTNESS_FLOOR, int(otsu_thr) - CARD_OTSU_MARGIN)
    bright = cv2.inRange(v, threshold, 255)
    low_sat = cv2.inRange(s, 0, CARD_SATURATION_MAX)
    mask = cv2.bitwise_and(bright, low_sat)
    # Drop the upper half entirely so backdrop glare never reaches findContours.
    mask[:roi_top, :] = 0

    def remove_long_horizontal_runs(src):
        # Zero out any row whose longest run of contiguous white pixels exceeds
        # the limit (a bright full-width table/edge strip). Fully vectorized:
        # for each cell, the length of the white run ending there is the column
        # index minus the index of the last black cell at or before it.
        out = src.copy()
        h, w = out.shape
        max_run_limit = int(w * CARD_ROW_RUN_REJECT_FRAC)
        b = out > 0
        cols = np.arange(w)
        zero_pos = np.where(~b, cols[None, :], -1)
        last_zero = np.maximum.accumulate(zero_pos, axis=1)
        run_len = np.where(b, cols[None, :] - last_zero, 0)
        bad_rows = run_len.max(axis=1) > max_run_limit
        out[bad_rows, :] = 0
        return out

    # The setup often has a bright horizontal table/edge strip under the cards.
    # Remove rows with a single long continuous white run. Five separate cards
    # can still produce many white pixels in a row, but not one full-width run.
    mask = remove_long_horizontal_runs(mask)

    k = max(3, int(CARD_CLOSE_KERNEL) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    mask = remove_long_horizontal_runs(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return mask


# ============================================================
# approach (floor placement) — per-card perspective rectification
#
# When the cards lie flat on the floor and the fixed forward camera views them
# at an angle, each card images as a trapezoid (keystone). The corners below let
# us warp every card back to a head-on rectangle so the whole downstream
# comparison pipeline (prepare_crop, multiscale, center-fine, ROTATE_180) keeps
# working unchanged.
# ============================================================

def _order_quad(pts):
    # Order 4 points as tl, tr, br, bl. The old x+y / x-y sum-and-diff logic
    # fails on cards seen under heavy foreshortening: when the card images as a
    # very wide, short trapezoid (e.g. 300px wide, 100px tall), x dominates both
    # the sum and the diff, so the same corner can win two slots and a corner is
    # duplicated -- collapsing the warp to a degenerate (often black) output that
    # fails _warp_is_valid. Instead, split by Y into a top pair and a bottom
    # pair, then sort each pair by X. This stays correct for any non-vertical
    # keystone regardless of aspect ratio.
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    ys = np.argsort(pts[:, 1])          # ascending by Y
    top = pts[ys[:2]]                   # two smallest Y -> top edge
    bottom = pts[ys[2:]]                # two largest Y  -> bottom edge
    tl, tr = top[np.argsort(top[:, 0])]        # sort top by X -> left, right
    bl, br = bottom[np.argsort(bottom[:, 0])]  # sort bottom by X -> left, right
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _card_quad(contour):
    # Recover the 4 outer corners of the (perspective-distorted) card. On the
    # floor the real contour is a trapezoid, so approxPolyDP captures the true
    # quad. At an extreme grazing angle the keystoned edges go jagged/pixelated
    # and a single fixed epsilon returns 6-8 points, forcing a minAreaRect
    # fallback that only yields a rectangle (no keystone) and breaks rectify.
    # Sweep epsilon upward until the simplification collapses to exactly 4
    # corners, keeping the true trapezoid; minAreaRect stays as last resort.
    peri = cv2.arcLength(contour, True)
    for eps in np.linspace(0.01, 0.15, 15):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return _order_quad(approx.reshape(-1, 2))
    return _order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))


def _quad_is_usable(quad):
    # Reject a degenerate quad before warping. At an extreme grazing angle the
    # recovered corners can collapse to a sliver that warps to a near-blank crop.
    # Measure fill against the quad's OWN min-area (rotated) rectangle, NOT the
    # axis-aligned bounding box: a legitimately slanted edge-card quad fills only
    # a small fraction of its axis-aligned bbox (the bbox of a rotated rectangle
    # is far larger than the rectangle itself), so a bbox-based test systematically
    # rejected exactly the grazing edge cards we most need to rectify and dropped
    # them to the un-rectified, slanted axis-aligned fallback crop. The rotated-rect
    # fill is rotation-invariant: a real card quad fills it ~0.8+ at any slant,
    # while only a collapsed sliver falls below the floor.
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    quad_area = abs(cv2.contourArea(q))
    (_, _), (rw, rh), _ = cv2.minAreaRect(q)
    rect_area = float(max(rw * rh, 1.0))
    return (quad_area / rect_area) >= CARD_QUAD_MIN_FILL


def _candidate_from_contour(frame_shape, contour):
    h, w = frame_shape[:2]
    frame_area = float(max(w * h, 1))
    x, y, bw, bh = cv2.boundingRect(contour)

    # approach 3 — reject blobs hugging the TOP edge only: that is the backdrop
    # wall / glare, which always reaches the top of the frame. The left, right and
    # bottom edges are NOT filtered — after the 180 deg turn-back the framing
    # drifts and an outer floor card can legitimately touch a side or bottom
    # border. Side junk is instead dropped by the row clustering (_select_row),
    # which keeps the densest horizontal band of card-like blobs.
    m = CARD_EDGE_MARGIN_PX
    if y <= m:
        return None, f"edge_touch_top y={y}"

    # approach 3 — floor ROI: a blob centered in the upper band is backdrop glare,
    # never a floor card (which images low in the frame). Center-based so it also
    # rejects glare in the middle of the wall that does not touch the top edge.
    if (y + bh / 2.0) < (h * CARD_ROI_TOP_FRAC):
        return None, f"wall_roi cy={int(y + bh / 2.0)}"

    # approach 3 — measure geometry with minAreaRect, not the axis-aligned box.
    # A card tilted/foreshortened on the floor inflates its boundingRect area
    # and skews its width/height aspect; the min-area rotated rectangle reports
    # the card's true size and side ratio regardless of orientation.
    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect
    long_side = float(max(rw, rh, 1.0))
    short_side = float(max(min(rw, rh), 1.0))
    rect_area = float(max(rw * rh, 1.0))
    contour_area = float(cv2.contourArea(contour))
    area_frac = rect_area / frame_area
    aspect = short_side / long_side  # ~0.63 head-on; peaks toward 1.0 under foreshortening
    solidity = contour_area / rect_area

    if area_frac < CARD_MIN_AREA_FRAC:
        return None, f"too_small area_frac={area_frac:.4f}"
    if area_frac > CARD_MAX_AREA_FRAC:
        return None, f"too_large area_frac={area_frac:.4f}"
    if aspect < CARD_MIN_ASPECT:
        return None, f"bad_aspect aspect={aspect:.2f}"
    if aspect > CARD_MAX_ASPECT:
        return None, f"bad_aspect aspect={aspect:.2f}"
    if solidity < CARD_MIN_SOLIDITY:
        return None, f"low_solidity solidity={solidity:.2f}"

    # approach 3 — drop the vertical_score: cards on the floor are no longer
    # expected near the top of the frame, so only center/aspect/size/solidity
    # drive the score.
    cx = x + bw / 2.0
    center_score = 1.0 - min(1.0, abs(cx - w / 2.0) / (w / 2.0))
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.63) / 0.45)
    size_score = min(1.0, area_frac / 0.04)
    score = 0.34 * aspect_score + 0.30 * size_score + 0.24 * solidity + 0.12 * center_score
    quad = _card_quad(contour)
    rect_quad = _order_quad(cv2.boxPoints(rect))
    return CardCandidate(
        (x, y, bw, bh), contour_area, aspect, solidity, float(score),
        quad=quad, rect_quad=rect_quad,
    ), ""


def _select_row(candidates, frame_h):
    # Always cluster into the densest horizontal band, even with <= NUM_SLOTS
    # candidates. Floor cards share a near-identical cy; a door/wall blob that
    # slipped past the geometry gates sits at a different cy and forms its own
    # (smaller) band, so anchoring on the densest band drops it. Skipping the
    # clustering when few candidates were found was the hole that let a single
    # wall/door blob be accepted as a card.
    if len(candidates) <= 1:
        return candidates

    best = []
    best_score = -1.0
    for anchor in candidates:
        ay = anchor.bbox[1] + anchor.bbox[3] / 2.0
        row = [
            c for c in candidates
            if abs((c.bbox[1] + c.bbox[3] / 2.0) - ay) <= CARD_ROW_Y_TOL_FRAC * frame_h
        ]
        row.sort(key=lambda c: c.score, reverse=True)
        row = row[:NUM_SLOTS]
        score = len(row) * 10.0 + sum(c.score for c in row)
        if score > best_score:
            best = row
            best_score = score
    return best


def _nms(candidates):
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected = []
    for cand in ordered:
        if any(_bbox_iou(cand.bbox, old.bbox) > CARD_NMS_IOU for old in selected):
            continue
        selected.append(cand)
    return selected


_RECTIFY_DST = np.array(
    [[0, 0], [CROP_W - 1, 0], [CROP_W - 1, CROP_H - 1], [0, CROP_H - 1]],
    dtype=np.float32,
)


def _warp_quad(frame, quad):
    # Rectify the card by warping its 4 ordered corners to the head-on canonical
    # rectangle. MORPH_OPEN erodes the card border in the mask, so the recovered
    # quad sits a few px inside the real card edge — push the corners outward from
    # the centroid to recapture that border (where a card's asymmetric detail often
    # lives). Grazing-angle edge cards are short, foreshortened source regions that
    # the warp upscales heavily; INTER_CUBIC keeps the upscaled edges sharper than
    # the default INTER_LINEAR, which smears low-frequency detail into a blurry haze.
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    c = q.mean(axis=0, keepdims=True)
    q = c + (q - c) * (1.0 + CARD_QUAD_EXPAND)
    m = cv2.getPerspectiveTransform(q, _RECTIFY_DST)
    warped = cv2.warpPerspective(frame, m, (CROP_W, CROP_H), flags=cv2.INTER_CUBIC)
    return warped if warped.size else None


def _warp_is_valid(crop):
    # A degenerate or mis-ordered quad warps to a near-uniform patch of off-card
    # background (the blank olive crop that scored 1.0/1.0 == UNCLEAR for every
    # orientation). Real card faces — white pip cards AND colorful card backs alike
    # — always carry structure, so a grayscale std floor rejects the blank warp
    # without touching valid crops. (A white-pixel-fraction test would wrongly
    # reject the colorful faces, so std, not brightness, is the right signal.)
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) >= CARD_CROP_MIN_STD


def _crop(frame, cand):
    # approach 1 — warp the card to a head-on canonical rectangle so a card scanned
    # before/after a flip yields comparable, undistorted crops. Try the recovered
    # trapezoid quad first (it also undoes the floor keystone). At a grazing angle
    # the contour goes jagged and that quad can be degenerate/mis-ordered — warping
    # it then samples off-card background. So fall back to the min-area rotated
    # rectangle, which has clean corners and robustly deslants edge cards (no
    # keystone correction, but far better than the slanted axis-aligned crop). Each
    # warp is validated; the axis-aligned bbox crop is the last resort.
    for quad in (getattr(cand, "quad", None), getattr(cand, "rect_quad", None)):
        if quad is None or not _quad_is_usable(quad):
            continue
        warped = _warp_quad(frame, quad)
        if _warp_is_valid(warped):
            return warped

    h, w = frame.shape[:2]
    x, y, bw, bh = cand.bbox
    pad = max(4, int(min(bw, bh) * 0.04))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return crop
    # INTER_AREA only beats cubic when shrinking; a foreshortened card is usually
    # upscaled to the canonical canvas, where INTER_CUBIC is the sharper choice.
    upscaling = (x2 - x1) < CROP_W or (y2 - y1) < CROP_H
    interp = cv2.INTER_CUBIC if upscaling else cv2.INTER_AREA
    return cv2.resize(crop, (CROP_W, CROP_H), interpolation=interp)


def _draw_overlay(frame, accepted, rejected):
    overlay = frame.copy()
    for cand, reason in rejected:
        x, y, w, h = cand
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 80, 255), 1)
        cv2.putText(overlay, reason[:24], (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
    for idx, cand in enumerate(accepted, start=1):
        x, y, w, h = cand.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(overlay, f"Card {idx}", (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
    return overlay


def detect_cards(frame):
    if frame is None or frame.size == 0:
        empty = np.zeros((480, 640), dtype=np.uint8)
        return DetectionResult([], empty, cv2.cvtColor(empty, cv2.COLOR_GRAY2BGR), [], [], 0)

    mask = build_card_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Cap CPU: process only the largest-area contours. A real 5-card scene has a
    # handful; a noisy backdrop/floor mask can otherwise produce hundreds, and the
    # per-contour minAreaRect/approxPolyDP work was spiking CPU and hanging the Pi.
    contour_count = len(contours)
    if contour_count > CARD_MAX_CONTOURS:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:CARD_MAX_CONTOURS]

    candidates = []
    rejected = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cand, reason = _candidate_from_contour(frame.shape, contour)
        if cand is None:
            rejected.append(((x, y, w, h), reason))
        else:
            candidates.append(cand)

    # Track candidates that passed the gates but get dropped later by NMS or row
    # selection, so a "4/5 cards" failure is debuggable: these would otherwise be
    # invisible (neither accepted-green nor rejected-red) in the overlay/logs.
    # Identity (id) sets, not `in`: CardCandidate carries a numpy quad field, so
    # dataclass __eq__ would do an ambiguous element-wise array comparison.
    after_nms = _nms(candidates)
    nms_ids = {id(c) for c in after_nms}
    for c in candidates:
        if id(c) not in nms_ids:
            rejected.append((c.bbox, f"dropped_nms aspect={c.aspect:.2f} score={c.score:.2f}"))

    after_row = _select_row(after_nms, frame.shape[0])
    accepted = sorted(after_row, key=lambda c: c.bbox[0] + c.bbox[2] / 2.0)[:NUM_SLOTS]
    accepted_ids = {id(c) for c in accepted}
    for c in after_nms:
        if id(c) not in accepted_ids:
            cy = c.bbox[1] + c.bbox[3] / 2.0
            rejected.append((c.bbox, f"dropped_row cy={cy:.0f} score={c.score:.2f}"))

    crops = [_crop(frame, c) for c in accepted]
    overlay = _draw_overlay(frame, accepted, rejected)
    return DetectionResult(accepted, mask, overlay, crops, rejected, contour_count)
