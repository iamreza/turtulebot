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

from .config import *


class V2Logger:
    def __init__(self):
        self.lock = threading.Lock()
        os.makedirs(RUN_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    def log(self, message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        print(line, flush=True)
        with self.lock:
            with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def save_image(self, path, image):
        if image is None:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
        return path

    def save_debug_set(self, raw, mask, overlay, crops):
        paths = {}
        paths["raw"] = self.save_image(LATEST_RAW, raw)
        paths["mask"] = self.save_image(LATEST_MASK, mask)
        paths["overlay"] = self.save_image(LATEST_OVERLAY, overlay)
        crop_paths = []
        for idx, crop in enumerate(crops, start=1):
            path = os.path.join(RUN_DIR, f"card_{idx:02d}.png")
            crop_paths.append(self.save_image(path, crop))
        paths["crops"] = crop_paths
        return paths


logger = V2Logger()
