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

import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template_string, send_from_directory

from .config import *
from .detector import detect_cards
from .logger import logger
from .templates import PAGE
from .workflow import run_rotation_workflow


app = Flask(__name__)
state = None
node = None
_last_cpu_sample = None


def set_state(app_state):
    global state
    state = app_state


def set_node(camera_node):
    global node
    node = camera_node


@app.route("/")
def index():
    return render_template_string(PAGE)


def _read_cpu_percent():
    global _last_cpu_sample
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            parts = f.readline().split()
        values = [int(v) for v in parts[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        sample = (idle, total)
        if _last_cpu_sample is None:
            _last_cpu_sample = sample
            return None
        prev_idle, prev_total = _last_cpu_sample
        _last_cpu_sample = sample
        total_delta = total - prev_total
        if total_delta <= 0:
            return None
        return max(0, min(100, round((1.0 - (idle - prev_idle) / total_delta) * 100)))
    except Exception:
        return None


def _read_ram_percent():
    try:
        values = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, raw_value = line.split(":", 1)
                values[key] = int(raw_value.strip().split()[0])
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if not total or available is None:
            return None
        return max(0, min(100, round((1.0 - available / total) * 100)))
    except Exception:
        return None


def _read_battery_status():
    base = "/sys/class/power_supply"
    try:
        if not os.path.isdir(base):
            return {"percent": None, "state": "unknown"}
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            type_path = os.path.join(path, "type")
            try:
                with open(type_path, "r", encoding="utf-8") as f:
                    supply_type = f.read().strip().lower()
            except Exception:
                supply_type = ""
            if supply_type != "battery" and not name.upper().startswith("BAT"):
                continue

            percent = None
            status = "unknown"
            capacity_path = os.path.join(path, "capacity")
            status_path = os.path.join(path, "status")
            try:
                with open(capacity_path, "r", encoding="utf-8") as f:
                    percent = max(0, min(100, int(float(f.read().strip()))))
            except Exception:
                pass
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    status = f.read().strip().lower()
            except Exception:
                pass

            if status == "charging":
                state = "charging"
            elif percent is not None and percent <= 15:
                state = "low"
            elif percent is not None and percent <= 35:
                state = "mid"
            elif percent is not None:
                state = "ok"
            else:
                state = "unknown"
            return {"percent": percent, "state": state, "status": status, "source": name}
    except Exception:
        pass
    return {"percent": None, "state": "unknown"}


def get_system_metrics(camera_node=None):
    battery = None
    if camera_node is not None:
        try:
            battery = camera_node.battery_status()
        except Exception:
            battery = None
    if not battery or battery.get("percent") is None:
        battery = _read_battery_status()

    return {
        "cpu_percent": _read_cpu_percent(),
        "ram_percent": _read_ram_percent(),
        "battery": battery,
    }


def _blank_frame(text):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    return frame


def _jpg_response(frame):
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return jpg.tobytes() if ok else b""


# Phases where the camera faces the cards, so the detection boxes line up with
# the live scene. While the robot is turning/waiting (away, wait, return) the
# stale boxes would float over an unrelated view, so we skip drawing them.
_BOX_PHASES = {"baseline", "check", "done"}


def build_camera_feed_frame():
    # Always render the live frame; draw the most recent detection boxes on a
    # copy so the feed never freezes on a stale overlay snapshot.
    with state.lock:
        raw = state.latest_raw.copy() if state.latest_raw is not None else None
        boxes = list(state.latest_boxes) if state.ui_phase in _BOX_PHASES else []
    if raw is None:
        return _blank_frame("WAITING FOR CAMERA")
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(raw, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(raw, f"Card {idx}", (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
    return raw


def build_mask_feed_frame():
    # Compute detection on demand from the freshest raw frame so the debug view
    # always tracks the current scene (no stale/black mask). Instead of a flat
    # binary silhouette, show the actual card pixels *through* the mask so the
    # card detail is visible, then overlay accepted (green) / rejected (red)
    # boxes with their reject reasons to make the segmentation debuggable.
    raw = state.get_frame()
    if raw is None:
        return _blank_frame("WAITING FOR MASK")
    result = detect_cards(raw)
    debug = cv2.bitwise_and(raw, raw, mask=result.mask)
    for (x, y, w, h), reason in result.rejected:
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 80, 255), 1)
        cv2.putText(debug, reason[:24], (x, max(18, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
    for idx, cand in enumerate(result.cards, start=1):
        x, y, w, h = cand.bbox
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug, f"Card {idx}", (x, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return debug


def mjpeg_generator(frame_builder, label):
    while True:
        try:
            frame = frame_builder()
            payload = _jpg_response(frame)
            if payload:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        except Exception as exc:
            logger.log(f"{label} feed error: {exc}")
        time.sleep(0.08)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(build_camera_feed_frame, "camera"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/mask_feed")
def mask_feed():
    return Response(mjpeg_generator(build_mask_feed_frame, "mask"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logs/<path:filename>")
def serve_log_file(filename):
    abs_log_dir = os.path.abspath(LOG_DIR)
    target = os.path.abspath(os.path.join(abs_log_dir, filename))
    if not target.startswith(abs_log_dir) or not os.path.exists(target):
        abort(404)
    return send_from_directory(abs_log_dir, filename)


@app.route("/status")
def status():
    slots = state.slot_info()
    with state.lock:
        missing_refs = [i for i in range(1, NUM_SLOTS + 1) if state.reference_images[i] is None]
        missing_checks = [i for i in range(1, NUM_SLOTS + 1) if state.checks[i] is None]
        data = {
            "status": state.status_text,
            "ui_phase": state.ui_phase,
            "workflow_active": state.workflow_active,
            "result": state.result_text,
            "references": {i: state.reference_images[i] is not None for i in range(1, NUM_SLOTS + 1)},
            "checks": {i: state.checks[i] is not None for i in range(1, NUM_SLOTS + 1)},
            "missing_refs": missing_refs,
            "missing_checks": missing_checks,
            "last_report": state.last_detect_summary,
            "slots": slots,
            "system": get_system_metrics(node),
        }
    data["cmd_vel_subscribers"] = node.cmd_vel_subscriber_count() if node else 0
    data["base_alive"] = node.base_alive() if node else False
    data["base_health"] = "odom ok" if data["base_alive"] else "no /odom"
    return jsonify(data)


def _start_workflow():
    with state.lock:
        running = state.workflow_thread is not None and state.workflow_thread.is_alive()
    if running:
        return False, "Workflow already running"
    if node is None:
        return False, "Robot node not ready"
    if state.get_frame() is None:
        return False, "No camera frame available"

    thread = threading.Thread(
        target=run_rotation_workflow, args=(node, state, logger),
        name="RotationWorkflow", daemon=True,
    )
    with state.lock:
        state.workflow_thread = thread
    thread.start()
    return True, "Rotation workflow started"


@app.route("/action/<name>", methods=["POST"])
def action(name):
    t0 = time.time()
    try:
        if name == "start":
            ok, message = _start_workflow()
        elif name in ("clear_all", "stop_robot"):
            state.stop_event.set()
            if node is not None:
                node.publish_stop()
            state.clear()
            ok, message = True, "Stopped and cleared v2 state"
        else:
            ok, message = False, f"Unknown action: {name}"
    except Exception as exc:
        logger.log(f"action {name} failed: {exc}")
        ok, message = False, str(exc)

    logger.log(f"action={name} ok={ok} duration={(time.time() - t0) * 1000:.1f}ms message={message}")
    return jsonify({"ok": ok, "message": message})


def run_flask():
    app.run(host="0.0.0.0", port=PORT, threaded=True)
