import cv2, json, numpy as np

# Load homography and pitch
H = np.load("homography_matrix.npy")
pitch = cv2.imread("pitch_template.jpg")

# Load YOLO detection JSON
with open("video_01.json") as f:
    data = json.load(f)

print(f"✅ Total frames in JSON: {len(data)}")

# Prepare video writer
h, w, _ = pitch.shape
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("mapped_output_fixed.avi", fourcc, 15, (w, h))

# Loop over all frames
for i, frame in enumerate(data):
    pitch_frame = pitch.copy()
    objs = frame.get("objects", [])
    
    # Debug print every 100 frames
    if i % 100 == 0:
        print(f"Frame {i}: {len(objs)} objects")
    
    for obj in objs:
        if "bbox_xyxy" in obj:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt, H)
            x, y = mapped[0][0]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(pitch_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    out.write(pitch_frame)

out.release()
print("🎯 Done — check mapped_output_fixed.avi")
