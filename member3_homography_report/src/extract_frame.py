import cv2

# Path to your downloaded video
video_path = r"C:\Users\sumai\path\to\SoccerNet\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley\1_720p.mkv"

cap = cv2.VideoCapture(video_path)

# Select frame number — try 5000 or adjust if needed
cap.set(cv2.CAP_PROP_POS_FRAMES, 5000)

success, frame = cap.read()

if success:
    cv2.imwrite("sample_frame.jpg", frame)
    print("✅ Frame successfully extracted and saved as sample_frame.jpg")
else:
    print("❌ Could not read frame. Try a different frame index or check path.")

cap.release()
