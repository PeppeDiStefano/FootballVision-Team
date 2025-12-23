import numpy as np
import cv2
import os

# =========================================================
# PATH SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_DST_PATH = os.path.join(DATA_DIR, "pitch_template.jpg")
SRC_PTS_PATH = os.path.join(DATA_DIR, "src_points.npy")
DST_PTS_PATH = os.path.join(DATA_DIR, "dst_points.npy")
H_PATH = os.path.join(OUT_DIR, "homography_matrix.npy")

# =========================================================
# LOAD DATA
# =========================================================
pitch = cv2.imread(IMG_DST_PATH)
if pitch is None:
    raise FileNotFoundError("❌ pitch_template.jpg not found")

src_points = np.load(SRC_PTS_PATH)
dst_points = np.load(DST_PTS_PATH)
H = np.load(H_PATH)

# =========================================================
# PROJECT SOURCE POINTS
# =========================================================
src_pts_reshaped = src_points.reshape(-1, 1, 2)
projected_pts = cv2.perspectiveTransform(src_pts_reshaped, H)
projected_pts = projected_pts.reshape(-1, 2)

# =========================================================
# DRAW POINTS
# =========================================================
vis = pitch.copy()

for i, (proj, gt) in enumerate(zip(projected_pts, dst_points)):
    px, py = int(proj[0]), int(proj[1])
    gx, gy = int(gt[0]), int(gt[1])

    # projected point (red)
    cv2.circle(vis, (px, py), 6, (0, 0, 255), -1)
    # ground truth point (green)
    cv2.circle(vis, (gx, gy), 6, (0, 255, 0), 2)

    cv2.line(vis, (px, py), (gx, gy), (255, 0, 0), 2)
    cv2.putText(
        vis,
        str(i + 1),
        (gx + 5, gy - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

# =========================================================
# SHOW
# =========================================================
cv2.imshow("Homography Preview (Red=Projected, Green=GT)", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
