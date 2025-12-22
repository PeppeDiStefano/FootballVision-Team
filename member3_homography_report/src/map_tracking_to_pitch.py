import numpy as np
import cv2
import json
import os

# === Load required files ===
H = np.load("homography_matrix.npy")

pitch_img_base = cv2.imread("pitch_template.jpg")
if pitch_img_base is None:
    raise FileNotFoundError("❌ pitch_template.jpg not found or cannot be read")

with open("tracking_output.json") as f:
    tracking_data = json.load(f)

# === Verify data ===
print(f"Total frames in JSON: {len(tracking_data)}")
h, w, _ = pitch_img_base.shape
print(f"✅ Pitch template size: {w}x{h}")

# === Setup video writer ===
out_path = "tracked_topdown.avi"
fps = 10
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

frames_written = 0

# === Main loop ===
for frame_data in tracking_data:
    frame_id = frame_data.get("frame_id")
    objects = frame_data.get("tracked_objects", [])

    if not objects:
        continue

    frame = pitch_img_base.copy()

    for obj in objects:
        if obj.get("class_name") != "person":
            continue

        x1, y1, x2, y2 = obj["bbox_xyxy"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)[0][0]
        mx, my = int(mapped[0]), int(mapped[1])

        tid = int(obj["track_id"])
        color = (int(tid * 47) % 255, int(tid * 67) % 255, int(tid * 89) % 255)

        if 0 <= mx < w and 0 <= my < h:
            cv2.circle(frame, (mx, my), 6, color, -1)

    out.write(frame)
    frames_written += 1

out.release()
print(f"🎥 Done — video saved as '{out_path}' with {frames_written} frames.")
