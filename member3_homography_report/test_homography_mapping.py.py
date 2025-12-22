import cv2
import numpy as np

# Load images
img_src = cv2.imread("sample_frame.jpg")
img_dst = cv2.imread("pitch_template.jpg")

# Load the saved homography matrix
H = np.load("homography_matrix.npy")
print("Loaded homography matrix:\n", H)

# Define test points in the broadcast frame (near the visible field)
points_src = np.array([[[400, 600], [700, 500], [900, 450], [1200, 500]]], dtype=np.float32)

# Apply the homography
points_mapped = cv2.perspectiveTransform(points_src, H)
print("\nMapped coordinates on pitch:")
print(points_mapped)

# Draw red dots for visible points
for (x, y) in points_mapped[0]:
    if 0 <= x < img_dst.shape[1] and 0 <= y < img_dst.shape[0]:
        cv2.circle(img_dst, (int(x), int(y)), 8, (0, 0, 255), -1)
    else:
        print(f"⚠️  Point ({x:.2f}, {y:.2f}) is outside the visible pitch range")

# Show result
cv2.imshow("Mapped points on pitch", img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
