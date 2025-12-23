import cv2
import os

# =========================================================
# PATH SETUP (REPRODUCIBLE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)

VIDEO_PATH = os.path.join(
    REPO_DIR,
    "member1_detection",
    "data",
    "raw",
    "video_01.mp4"
)

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FRAME = os.path.join(DATA_DIR, "homography_frame.jpg")

# =========================================================
# LOAD VIDEO
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Cannot open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =========================================================
# SELECT CENTRAL FRAME
# =========================================================
frame_idx = total_frames // 2
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("❌ Failed to read frame from video")

# =========================================================
# SAVE FRAME
# =========================================================
cv2.imwrite(OUTPUT_FRAME, frame)

print("✅ Homography reference frame extracted")
print(f"   → Path: {OUTPUT_FRAME}")
print(f"   → Frame index: {frame_idx} / {total_frames}")
print(f"   → Resolution: {width} x {height}")