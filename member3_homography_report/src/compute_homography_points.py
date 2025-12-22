import cv2
import numpy as np

# === Step 1: Load your images ===
img_src = cv2.imread("sample_frame.jpg")     # broadcast frame
img_dst = cv2.imread("pitch_template.jpg")   # top-down field

# Resize pitch template if needed (match roughly similar width)
img_dst = cv2.resize(img_dst, (800, 500))

# === Step 2: Select matching points ===
# These are example pixel coordinates — you’ll manually adjust them.
# The order of points must correspond between both images.

# Points on your sample_frame (x, y)
src_points = np.array([
    [200, 600],   # bottom-left corner of field
    [1200, 580],  # bottom-right corner
    [800, 400],   # center circle top
    [250, 300]    # top-left penalty box corner
], dtype=np.float32)

# Points on your pitch_template (X, Y)
dst_points = np.array([
    [100, 450],
    [700, 450],
    [400, 250],
    [150, 150]
], dtype=np.float32)

# === Step 3: Compute homography matrix ===
H, _ = cv2.findHomography(src_points, dst_points)

# === Step 4: Warp broadcast image into top-down view ===
warped = cv2.warpPerspective(img_src, H, (img_dst.shape[1], img_dst.shape[0]))

# === Step 5: Save and show ===
cv2.imwrite("homography_result.jpg", warped)
print("✅ Homography applied — result saved as homography_result.jpg")
