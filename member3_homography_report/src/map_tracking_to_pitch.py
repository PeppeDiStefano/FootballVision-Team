import numpy as np
import cv2
import json
import os

# =========================================================
# PATH SETUP (REPRODUCIBLE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

TRACKING_JSON = os.path.join(
    REPO_DIR,
    "member2_tracking",
    "data",
    "outputs",
    "tracking_output.json"
)

H_PATH = os.path.join(OUT_DIR, "homography_matrix.npy")
PITCH_PATH = os.path.join(DATA_DIR, "pitch_template.jpg")

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
H = np.load(H_PATH)
pitch = cv2.imread(PITCH_PATH)
if pitch is None:
    raise FileNotFoundError("❌ pitch_template.jpg not found")

with open(TRACKING_JSON, "r") as f:
    tracking_data = json.load(f)

h, w, _ = pitch.shape
print(f"Tracking frames: {len(tracking_data)}")
print(f"Pitch size: {w} x {h}")

# =========================================================
# VIDEO WRITER
# =========================================================
out_path = os.path.join(OUT_DIR, "tracked_topdown.mp4")
fps = 10
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

# =========================================================
# MAIN LOOP
# =========================================================
for frame_data in tracking_data:
    frame = pitch.copy()
    objects = frame_data.get("tracked_objects", [])

    for obj in objects:
        if obj.get("class_name") != "person":
            continue

        x1, y1, x2, y2 = obj["bbox_xyxy"]

        # bottom-center of bbox
        cx = (x1 + x2) / 2.0
        cy = y2

        pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]

        mx, my = int(mapped[0]), int(mapped[1])

        tid = int(obj["track_id"])
        color = (
            (tid * 47) % 255,
            (tid * 67) % 255,
            (tid * 89) % 255
        )

        if 0 <= mx < w and 0 <= my < h:
            cv2.circle(frame, (mx, my), 6, color, -1)

    out.write(frame)

out.release()
print(f"\n🎥 DONE — Top-down video saved to:\n{out_path}")
