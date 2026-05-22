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
import time

import cv2

from .config import (
    BASELINE_SETTLE_SECONDS,
    DEMO_WAIT_SECONDS,
    MIN_GOOD_SCORE,
    NUM_SLOTS,
    ROBOT_TURN_ENABLED,
    ROTATION_MARGIN,
    RUN_DIR,
    SCAN_SETTLE_SECONDS,
    SHARPEST_FRAME_SECONDS,
    SLOT_TURN_SPEED,
)
from .detector import detect_cards
from .utils import (
    compare_center_fine_pair,
    compare_orientation_pair_multiscale,
    orientation_verdict,
    prepare_crop,
    prepare_crop_fine,
)


def _grab_sharpest_frame(state, logger, label, settle_s, sample_s):
    # Let the chassis settle, then sample distinct camera frames and keep the
    # one with the highest focus measure (variance of Laplacian). Motion blur
    # suppresses high-frequency detail, so the sharpest frame is the most
    # settled one -- this is what makes orientation matching reliable after a
    # turn. Returns the chosen BGR frame (or None if no frame ever arrives).
    if settle_s > 0:
        time.sleep(settle_s)

    best_frame, best_sharp, last_t, samples = None, -1.0, None, 0
    deadline = time.time() + max(0.0, sample_s)
    while True:
        frame, t = state.get_frame_with_time()
        if frame is not None and t != last_t:
            last_t = t
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            samples += 1
            if sharp > best_sharp:
                best_sharp, best_frame = sharp, frame
        if time.time() >= deadline:
            break
        time.sleep(0.03)

    logger.log(f"[{label}] sharpest frame: focus={best_sharp:.0f} from {samples} sample(s)")
    return best_frame


def _scan(node, state, logger, label, frame=None):
    # Run one detection pass. Uses the supplied frame when given (e.g. the
    # sharpest-frame pick after a turn), otherwise the freshest camera frame.
    # Returns (ok, result, crops) where crops is the list of BGR card crops.
    if frame is None:
        frame = state.get_frame()
    if frame is None:
        return False, None, "No camera frame available"

    result = detect_cards(frame)
    state.set_boxes([c.bbox for c in result.cards])
    summary = (
        f"[{label}] contours={result.contour_count} "
        f"accepted={len(result.cards)}/{NUM_SLOTS} rejected={len(result.rejected)}"
    )
    logger.log(summary)
    state.last_detect_summary = summary

    if len(result.cards) != NUM_SLOTS:
        return False, result, f"{label} scan saw {len(result.cards)}/{NUM_SLOTS} cards"
    return True, result, summary


def _cross_match(references, check_prepared):
    # Greedily assign each check crop to its best-scoring reference slot.
    # Returns {ref_slot: (check_index, score_normal, score_rot180)}.
    pairs = {}
    for ci, cp in enumerate(check_prepared):
        for s in range(1, NUM_SLOTS + 1):
            rp = references.get(s)
            if rp is None or cp is None:
                continue
            n, r = compare_orientation_pair_multiscale(rp, cp)
            pairs[(ci, s)] = (n, r)

    order = sorted(pairs.items(), key=lambda kv: max(kv[1]), reverse=True)
    assigned_check, assigned_ref, assign = set(), set(), {}
    for (ci, s), (n, r) in order:
        if ci in assigned_check or s in assigned_ref:
            continue
        assigned_check.add(ci)
        assigned_ref.add(s)
        assign[s] = (ci, n, r)
    return assign


def run_rotation_workflow(node, state, logger):
    state.stop_event.clear()
    with state.lock:
        state.workflow_active = True
        state.references = {i: None for i in range(1, NUM_SLOTS + 1)}
        state.reference_images = {i: None for i in range(1, NUM_SLOTS + 1)}
        state.check_images = {i: None for i in range(1, NUM_SLOTS + 1)}
        state.checks = {i: None for i in range(1, NUM_SLOTS + 1)}
        state.result_text = "0/5 detected"

    def fail(msg):
        # Terminal status must NOT contain "WORKFLOW_"/"WAIT" or the UI keeps
        # showing "Running" (see inferRunning in templates.py).
        logger.log(f"workflow FAILED: {msg}")
        with state.lock:
            state.workflow_active = False
            state.status_text = f"FAILED - {msg}"
            state.result_text = msg
        return False, msg

    def stopped():
        return state.stop_event.is_set()

    try:
        # ---- Phase 1: baseline scan (references) ----
        state.set_phase("baseline", "WORKFLOW_BASELINE - scanning references")
        base_frame = _grab_sharpest_frame(
            state, logger, "BASELINE", BASELINE_SETTLE_SECONDS, SHARPEST_FRAME_SECONDS
        )
        ok, result, msg = _scan(node, state, logger, "BASELINE", frame=base_frame)
        if not ok:
            return fail(msg)

        prepared, ref_paths, references_fine = {}, {}, {}
        for idx, crop in enumerate(result.crops, start=1):
            path = os.path.join(RUN_DIR, f"ref_{idx:02d}.png")
            ref_paths[idx] = logger.save_image(path, crop)
            prepared[idx] = prepare_crop(crop)
            references_fine[idx] = prepare_crop_fine(crop)
        state.set_references(prepared, ref_paths)
        with state.lock:
            state.result_text = f"{NUM_SLOTS}/{NUM_SLOTS} references captured"
        logger.log("workflow: 5 references captured")
        if stopped():
            return fail("stopped")

        # ---- Phase 2: turn 180 away (stage 4 — disabled while ROBOT_TURN_ENABLED is False) ----
        if ROBOT_TURN_ENABLED:
            state.set_phase("away", "WORKFLOW_LOOK_AWAY - turning 180 deg")
            ok, msg = node.turn_by_angle_blocking(+math.pi, SLOT_TURN_SPEED, "LOOK_AWAY_180", state.stop_event)
            if not ok:
                return fail(f"look-away turn failed: {msg}")
        else:
            logger.log("workflow: robot turn disabled (manual-flip mode) — skipping look-away")

        # ---- Phase 3: wait for human to flip a card ----
        state.set_phase("wait", f"WORKFLOW_WAIT - flip a card now ({int(DEMO_WAIT_SECONDS)}s)")
        deadline = time.time() + DEMO_WAIT_SECONDS
        while time.time() < deadline:
            if stopped():
                return fail("stopped")
            remaining = int(deadline - time.time()) + 1
            with state.lock:
                state.status_text = f"WORKFLOW_WAIT - flip a card now ({remaining}s)"
            time.sleep(0.2)

        # ---- Phase 4: turn 180 back (stage 4 — disabled while ROBOT_TURN_ENABLED is False) ----
        if ROBOT_TURN_ENABLED:
            state.set_phase("return", "WORKFLOW_LOOK_BACK - turning 180 deg back")
            ok, msg = node.turn_by_angle_blocking(-math.pi, SLOT_TURN_SPEED, "LOOK_BACK_180", state.stop_event)
            if not ok:
                return fail(f"look-back turn failed: {msg}")
        else:
            logger.log("workflow: robot turn disabled (manual-flip mode) — skipping look-back")

        # ---- Phase 5: check scan ----
        # Settle longer here: the turn-back just finished and the chassis is
        # still drifting (~7-10 deg of coast). Grabbing immediately is what
        # produced the motion-blurred CHECK crops that broke detection.
        state.set_phase("check", "WORKFLOW_CHECK - rescanning cards")
        check_frame = _grab_sharpest_frame(
            state, logger, "CHECK", SCAN_SETTLE_SECONDS, SHARPEST_FRAME_SECONDS
        )
        ok, result, msg = _scan(node, state, logger, "CHECK", frame=check_frame)
        if not ok:
            return fail(msg)

        check_prepared, check_prepared_fine, check_paths = [], [], {}
        for idx, crop in enumerate(result.crops, start=1):
            path = os.path.join(RUN_DIR, f"chk_{idx:02d}.png")
            check_paths[idx] = logger.save_image(path, crop)  # keyed by detection order
            check_prepared.append(prepare_crop(crop))
            check_prepared_fine.append(prepare_crop_fine(crop))

        # ---- Phase 6: compare (cross-match + orientation verdict) ----
        with state.lock:
            references = dict(state.references)
        assign = _cross_match(references, check_prepared)

        verdicts, slot_check_images, rotated_slots = {}, {}, []
        for s in range(1, NUM_SLOTS + 1):
            if s not in assign:
                continue
            ci, n, r = assign[s]
            n_fine, r_fine = compare_center_fine_pair(
                references_fine.get(s), check_prepared_fine[ci]
            )
            status = orientation_verdict(
                n, r, MIN_GOOD_SCORE, ROTATION_MARGIN,
                fine_normal=n_fine, fine_rot180=r_fine,
            )
            is_rotated = status == "ROTATED_180"
            verdicts[s] = {
                "orientation": "rot180" if is_rotated else "normal",
                "is_rotated": is_rotated,
                "status": status,
                "score_normal": n,
                "score_rot180": r,
                "score_fine_normal": n_fine,
                "score_fine_rot180": r_fine,
            }
            slot_check_images[s] = check_paths.get(ci + 1)
            if is_rotated:
                rotated_slots.append(s)
            logger.log(
                f"slot {s}: {status} | score_normal={n:.3f} score_rot180={r:.3f} "
                f"fine_normal={n_fine:.3f} fine_rot180={r_fine:.3f} "
                f"(check #{ci + 1})"
            )

        state.set_checks(slot_check_images, verdicts)

        # ---- Phase 7: done ----
        if rotated_slots:
            names = ", ".join(f"Card {s}" for s in sorted(rotated_slots))
            result_text = f"{names} rotated 180 deg"
        else:
            result_text = "No rotated card detected"
        with state.lock:
            state.result_text = result_text
            state.workflow_active = False
        # "DONE" (no WORKFLOW_ prefix) so the UI shows Complete, not Running.
        state.set_phase("done", f"DONE - {result_text}")
        logger.log(f"workflow DONE: {result_text}")
        return True, result_text

    except Exception as exc:
        return fail(f"exception: {exc}")
    finally:
        with state.lock:
            state.workflow_active = False
