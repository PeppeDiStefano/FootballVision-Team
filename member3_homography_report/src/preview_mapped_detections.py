import cv2
import json
import numpy as np

# Load your homography matrix and pitch
H = np.load("homography_matrix.npy")
pitch = cv2.imread("pitch_template.jpg")

# Load YOLO detection JSON
with open("video_01.json", "r") as f:
    data = json.load(f)

# Loop through frames (only first few for preview)
for frame in data[:10]:
    for obj in frame["objects"]:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cls = obj["class_name"]
        
        # Apply homography
        src_point = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(src_point, H)
        x, y = mapped[0][0]
        
        # Draw on pitch if inside image bounds
        if 0 <= x < pitch.shape[1] and 0 <= y < pitch.shape[0]:
            color = (0, 0, 255) if cls == "sports ball" else (0, 255, 0)
            cv2.circle(pitch, (int(x), int(y)), 5, color, -1)

# Show result
cv2.imshow("Projected YOLO detections on pitch", pitch)
cv2.waitKey(0)
cv2.destroyAllWindows()
