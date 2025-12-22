import cv2
import numpy as np

img_src = cv2.imread("sample_frame.jpg")
img_dst = cv2.imread("pitch_template.jpg")

# Example of your original matching points
src_points = np.array([[26,703], [1266,689], [266,227], [622,390]], dtype=np.float32)
dst_points = np.array([[57,389], [744,388], [399,41], [399,220]], dtype=np.float32)

H, _ = cv2.findHomography(src_points, dst_points)
print("Homography matrix computed:\n", H)

# Optional: save it for next time
np.save("homography_matrix.npy", H)
