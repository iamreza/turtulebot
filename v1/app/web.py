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
import time

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template_string, send_from_directory

from .config import *
from .templates import PAGE

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
robot_node = None
_last_cpu_sample = None


def set_robot_node(node):
    global robot_node
    robot_node = node


@app.route("/")
def index():
    return render_template_string(PAGE)


def get_missing_refs_and_checks():
    global robot_node

    with robot_node.lock:
        missing_refs = [
            i for i in range(1, NUM_SLOTS + 1)
            if robot_node.references[i] is None
        ]

        missing_checks = [
            i for i in range(1, NUM_SLOTS + 1)
            if robot_node.checks[i] is None
        ]

    return missing_refs, missing_checks


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
        idle_delta = idle - prev_idle
        total_delta = total - prev_total
        if total_delta <= 0:
            return None
        return max(0, min(100, round((1.0 - idle_delta / total_delta) * 100)))
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


def get_system_metrics(node=None):
    battery = None
    if node is not None:
        try:
            battery = node.battery_status()
        except Exception:
            battery = None
    if not battery or battery.get("percent") is None:
        battery = _read_battery_status()

    return {
        "cpu_percent": _read_cpu_percent(),
        "ram_percent": _read_ram_percent(),
        "battery": battery,
    }


@app.route("/status")
def route_status():
    global robot_node

    with robot_node.lock:
        refs = {
            i: robot_node.references[i] is not None
            for i in range(1, NUM_SLOTS + 1)
        }

        checks = {}
        for i in range(1, NUM_SLOTS + 1):
            c = robot_node.checks[i]
            if c is None:
                checks[i] = None
            else:
                checks[i] = {
                    "status": c["status"],
                    "score_normal": round(float(c["score_normal"]), 3),
                    "score_rot180": round(float(c["score_rot180"]), 3),
                    "time": c["time"],
                    "image": c["image"],
                }

        missing_refs, missing_checks = get_missing_refs_and_checks()

        data = {
            "status": robot_node.status_text,
            "ui_phase": robot_node.ui_phase,
            "workflow_active": robot_node.workflow_active,
            "result": robot_node.result_text,
            "references": refs,
            "checks": checks,
            "missing_refs": missing_refs,
            "missing_checks": missing_checks,
            "last_report": robot_node.last_report_path,
            "two_step_last_view": robot_node.two_step_last_view,
            "one_shot_last_view": robot_node.one_shot_last_view,
            "cmd_vel_subscribers": robot_node.cmd_vel_subscriber_count(),
            "base_alive": robot_node.base_alive(),
            "base_health": robot_node.base_health_text(),
            "slots": robot_node.get_slot_panel_info(),
            "system": get_system_metrics(robot_node),
        }

    return jsonify(data)


def _blank_frame(text):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        text,
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return frame


def build_camera_feed_frame():
    with robot_node.lock:
        raw = robot_node.latest_raw.copy() if robot_node.latest_raw is not None else None
        bbox = robot_node.latest_bbox

    if raw is None:
        return _blank_frame("WAITING FOR CAMERA")

    frame = raw.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(frame, (x + w // 2, y + h // 2), 6, (0, 0, 255), -1)
    return frame


def build_mask_feed_frame():
    with robot_node.lock:
        mask = robot_node.latest_mask.copy() if robot_node.latest_mask is not None else None

    if mask is None:
        return _blank_frame("WAITING FOR MASK")

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def mjpeg_generator(frame_builder, error_label):
    global robot_node

    while True:
        try:
            frame = frame_builder()
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpg.tobytes() +
                    b"\r\n"
                )

        except Exception as e:
            print(f"[{error_label} ERROR]", e, flush=True)

        time.sleep(0.08)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(build_camera_feed_frame, "VIDEO"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/mask_feed")
def mask_feed():
    return Response(
        mjpeg_generator(build_mask_feed_frame, "MASK VIDEO"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/logs/<path:filename>")
def serve_log_file(filename):
    # Serve saved reference / check / debug JPGs from logs/ so the UI can
    # display them. Restrict to files that already exist inside LOG_DIR
    # (Flask's send_from_directory already prevents path-traversal).
    abs_log_dir = os.path.abspath(LOG_DIR)
    target = os.path.abspath(os.path.join(abs_log_dir, filename))
    if not target.startswith(abs_log_dir):
        abort(404)
    if not os.path.exists(target):
        abort(404)
    return send_from_directory(abs_log_dir, filename)


@app.route("/action/<name>", methods=["POST"])
def action(name):
    global robot_node

    t0 = time.time()
    robot_node.trace("HTTP", f"Action received: {name}")

    ok = True
    message = "OK"

    try:
        if name == "start":
            ok, message = robot_node.start_workflow("FULL_AUTO_ONE_SCAN")

        elif name.startswith("save_slot_"):
            slot_id = int(name.split("_")[-1])
            ok, message = robot_node.save_slot(slot_id)

        elif name.startswith("check_slot_"):
            slot_id = int(name.split("_")[-1])
            ok, message = robot_node.check_slot(slot_id)

        elif name == "turn_left_small":
            ok, message = robot_node.start_motion(
                0.0,
                +TURN_SPEED,
                TURN_SMALL,
                "TURN_LEFT_SMALL"
            )

        elif name == "turn_left_medium":
            ok, message = robot_node.start_motion(
                0.0,
                +TURN_SPEED,
                TURN_MEDIUM,
                "TURN_LEFT_MEDIUM"
            )

        elif name == "turn_right_small":
            ok, message = robot_node.start_motion(
                0.0,
                -TURN_SPEED,
                TURN_SMALL,
                "TURN_RIGHT_SMALL"
            )

        elif name == "turn_right_medium":
            ok, message = robot_node.start_motion(
                0.0,
                -TURN_SPEED,
                TURN_MEDIUM,
                "TURN_RIGHT_MEDIUM"
            )

        elif name == "turn_next":
            ok, message = robot_node.start_motion(
                0.0,
                +TURN_SPEED,
                TURN_NEXT_DURATION,
                "TURN_NEXT"
            )

        elif name == "turn_previous":
            ok, message = robot_node.start_motion(
                0.0,
                -TURN_SPEED,
                TURN_NEXT_DURATION,
                "TURN_PREVIOUS"
            )

        elif name == "arc_next":
            ok, message = robot_node.start_motion(
                ARC_LINEAR_SPEED,
                +ARC_TURN_SPEED,
                ARC_NEXT_DURATION,
                "ARC_NEXT"
            )

        elif name == "arc_previous":
            ok, message = robot_node.start_motion(
                ARC_LINEAR_SPEED,
                -ARC_TURN_SPEED,
                ARC_NEXT_DURATION,
                "ARC_PREVIOUS"
            )

        elif name == "look_away_right":
            ok, message = robot_node.start_motion(
                0.0,
                -TURN_SPEED,
                LOOK_AWAY_RIGHT_DURATION,
                "LOOK_AWAY_RIGHT"
            )

        elif name == "look_away_left":
            ok, message = robot_node.start_motion(
                0.0,
                +TURN_SPEED,
                LOOK_AWAY_LEFT_DURATION,
                "LOOK_AWAY_LEFT"
            )

        elif name == "auto_save_all":
            ok, message = robot_node.start_workflow("AUTO_SAVE_ALL")

        elif name == "auto_check_all":
            ok, message = robot_node.start_workflow("AUTO_CHECK_ALL")

        elif name == "auto_return_to_slot_1":
            ok, message = robot_node.start_workflow("AUTO_RETURN_TO_SLOT_1")

        elif name == "step_next_slot":
            ok, message = robot_node.start_workflow("STEP_NEXT_SLOT")

        elif name == "step_previous_slot":
            ok, message = robot_node.start_workflow("STEP_PREVIOUS_SLOT")

        elif name == "auto_look_away_right":
            ok, message = robot_node.start_workflow("AUTO_LOOK_AWAY_RIGHT")

        elif name == "auto_look_back_left":
            ok, message = robot_node.start_workflow("AUTO_LOOK_BACK_LEFT")

        elif name == "start_scan":
            # Manual mode: each click scans one slot (save or check, depending
            # on phase). No robot motion — user moves the robot themselves.
            ok, message = robot_node.scan_current_slot()

        elif name == "one_shot_scan":
            # Best movement-minimal mode: baseline all 5 in one frame, wait, check all 5 in one frame.
            ok, message = robot_node.one_shot_scan_current_view()

        elif name == "two_step_scan":
            # Backup movement-minimal mode: baseline A, baseline B, wait, check A, check B.
            ok, message = robot_node.two_step_scan_current_view()

        elif name == "auto_run":
            # Fully autonomous: closed-loop motion + vision align between
            # slots, save 1->5, 90° away, wait 15s, 90° back, check 5->1.
            ok, message = robot_node.start_workflow("AUTO_RUN")

        elif name == "stop_robot":
            ok, message = robot_node.stop_robot()

        elif name == "clear_all":
            ok, message = robot_node.clear_all()

        else:
            ok = False
            message = f"Unknown action: {name}"
            robot_node.add_event(message)

    except Exception as e:
        ok = False
        message = f"Action error for {name}: {e}"
        print("[ACTION ERROR]", message, flush=True)

        try:
            robot_node.trace_exc("HTTP_ERR", f"Action {name}")
            robot_node.add_event(message)
            with robot_node.lock:
                robot_node.status_text = "ACTION_ERROR"
        except Exception:
            pass

    duration_ms = (time.time() - t0) * 1000.0
    robot_node.trace(
        "HTTP",
        f"Action completed: {name} | ok={ok} | duration={duration_ms:.1f}ms | message={message}"
    )

    return jsonify({
        "ok": ok,
        "message": message
    })
