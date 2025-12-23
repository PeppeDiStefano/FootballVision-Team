import cv2
import numpy as np
import os

# =========================================================
# PATH SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

IMG_SRC_PATH = os.path.join(DATA_DIR, "homography_frame.jpg")
IMG_DST_PATH = os.path.join(DATA_DIR, "pitch_template.jpg")

SRC_PTS_PATH = os.path.join(DATA_DIR, "src_points.npy")
DST_PTS_PATH = os.path.join(DATA_DIR, "dst_points.npy")

# =========================================================
# LOAD IMAGES
# =========================================================
img_src = cv2.imread(IMG_SRC_PATH)
img_dst = cv2.imread(IMG_DST_PATH)

if img_src is None:
    raise FileNotFoundError("❌ homography_frame.jpg not found")
if img_dst is None:
    raise FileNotFoundError("❌ pitch_template.jpg not found")

src_points = []
dst_points = []

# =========================================================
# MOUSE CALLBACK
# =========================================================
def click_event_src(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        src_points.append([x, y])
        cv2.circle(img_src, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            img_src,
            str(len(src_points)),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        cv2.imshow("SOURCE FRAME", img_src)

def click_event_dst(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        dst_points.append([x, y])
        cv2.circle(img_dst, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(
            img_dst,
            str(len(dst_points)),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
        cv2.imshow("PITCH TEMPLATE", img_dst)

# =========================================================
# SOURCE IMAGE SELECTION
# =========================================================
cv2.namedWindow("SOURCE FRAME", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("SOURCE FRAME", click_event_src)
cv2.imshow("SOURCE FRAME", img_src)

print("👉 Click points on SOURCE FRAME (press 'q' when done)")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyWindow("SOURCE FRAME")

# =========================================================
# DESTINATION IMAGE SELECTION
# =========================================================
cv2.namedWindow("PITCH TEMPLATE", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("PITCH TEMPLATE", click_event_dst)
cv2.imshow("PITCH TEMPLATE", img_dst)

print("👉 Click corresponding points on PITCH TEMPLATE (press 'q' when done)")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyWindow("PITCH TEMPLATE")

# =========================================================
# SAVE POINTS
# =========================================================
src_points = np.array(src_points, dtype=np.float32)
dst_points = np.array(dst_points, dtype=np.float32)

if len(src_points) != len(dst_points) or len(src_points) < 4:
    raise ValueError("❌ Number of points mismatch or too few points")

np.save(SRC_PTS_PATH, src_points)
np.save(DST_PTS_PATH, dst_points)

print(f"✅ Saved {len(src_points)} point correspondences")
print(f"   → {SRC_PTS_PATH}")
print(f"   → {DST_PTS_PATH}")