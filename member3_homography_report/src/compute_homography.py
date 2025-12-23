import numpy as np
import cv2
import os

# =========================================================
# PATH SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

SRC_PTS_PATH = os.path.join(DATA_DIR, "src_points.npy")
DST_PTS_PATH = os.path.join(DATA_DIR, "dst_points.npy")
H_PATH = os.path.join(OUT_DIR, "homography_matrix.npy")

# =========================================================
# LOAD POINTS
# =========================================================
src_points = np.load(SRC_PTS_PATH)
dst_points = np.load(DST_PTS_PATH)

if src_points.shape != dst_points.shape:
    raise ValueError("❌ src_points and dst_points shape mismatch")

if src_points.shape[0] < 4:
    raise ValueError("❌ At least 4 point correspondences required")

print(f"Loaded {src_points.shape[0]} point correspondences")

# =========================================================
# COMPUTE HOMOGRAPHY (RANSAC)
# =========================================================
H, mask = cv2.findHomography(
    src_points,
    dst_points,
    method=cv2.RANSAC,
    ransacReprojThreshold=5.0
)

if H is None:
    raise RuntimeError("❌ Homography computation failed")

np.save(H_PATH, H)

inliers = int(mask.sum()) if mask is not None else src_points.shape[0]

print("✅ Homography matrix computed")
print(f"Inliers: {inliers} / {src_points.shape[0]}")
print("Homography matrix:")
print(H)
print(f"Saved to: {H_PATH}")
