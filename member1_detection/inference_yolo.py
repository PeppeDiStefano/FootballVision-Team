from ultralytics import YOLO
import os

#paths
VIDEO_PATH = "data/video_01.mp4"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

#run inference on video
model(
    VIDEO_PATH,
    save=True,
    project=OUTPUT_DIR,
    name="baseline_yolo",
    conf=0.25
)

print("Inference completed.")