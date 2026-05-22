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

    otsu_thr, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(CARD_BRIGHTNESS_FLOOR, int(otsu_thr) - CARD_OTSU_MARGIN)
    bright = cv2.inRange(v, threshold, 255)
    low_sat = cv2.inRange(s, 0, CARD_SATURATION_MAX)
    mask = cv2.bitwise_and(bright, low_sat)

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


def _candidate_from_contour(frame_shape, contour):
    h, w = frame_shape[:2]
    frame_area = float(max(w * h, 1))
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = float(max(bw * bh, 1))
    contour_area = float(cv2.contourArea(contour))
    area_frac = bbox_area / frame_area
    aspect = bw / float(max(bh, 1))
    solidity = contour_area / bbox_area

    if area_frac < CARD_MIN_AREA_FRAC:
        return None, f"too_small area_frac={area_frac:.4f}"
    if area_frac > CARD_MAX_AREA_FRAC:
        return None, f"too_large area_frac={area_frac:.4f}"
    if aspect < CARD_MIN_ASPECT:
        return None, f"too_narrow aspect={aspect:.2f}"
    if aspect > CARD_MAX_ASPECT:
        return None, f"too_wide aspect={aspect:.2f}"
    if solidity < CARD_MIN_SOLIDITY:
        return None, f"low_solidity solidity={solidity:.2f}"

    cx = x + bw / 2.0
    cy = y + bh / 2.0
    center_score = 1.0 - min(1.0, abs(cx - w / 2.0) / (w / 2.0))
    vertical_score = 1.0 - min(1.0, cy / float(max(h, 1)))
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.63) / 0.35)
    size_score = min(1.0, area_frac / 0.04)
    score = 0.34 * aspect_score + 0.30 * size_score + 0.20 * solidity + 0.10 * center_score + 0.06 * vertical_score
    return CardCandidate((x, y, bw, bh), contour_area, aspect, solidity, float(score)), ""


def _select_row(candidates, frame_h):
    if len(candidates) <= NUM_SLOTS:
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


def _crop(frame, bbox):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    pad = max(4, int(min(bw, bh) * 0.04))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return crop
    return cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


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

    accepted = []
    rejected = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cand, reason = _candidate_from_contour(frame.shape, contour)
        if cand is None:
            rejected.append(((x, y, w, h), reason))
        else:
            accepted.append(cand)

    accepted = _nms(accepted)
    accepted = _select_row(accepted, frame.shape[0])
    accepted = sorted(accepted, key=lambda c: c.bbox[0] + c.bbox[2] / 2.0)[:NUM_SLOTS]

    crops = [_crop(frame, c.bbox) for c in accepted]
    overlay = _draw_overlay(frame, accepted, rejected)
    return DetectionResult(accepted, mask, overlay, crops, rejected, len(contours))
