from ultralytics import YOLO
import json
import os

VIDEO_PATH = "data/raw/video_01.mp4"
OUTPUT_DIR = "outputs/detections"
OUTPUT_JSON = "video_01.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

results = model(VIDEO_PATH, stream=True)

all_frames = []

for frame_idx, r in enumerate(results):
    frame_data = {
        "frame_id": frame_idx,
        "objects": []
    }

    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            frame_data["objects"].append({
                "class_id": int(cls),
                "class_name": model.names[int(cls)],
                "bbox_xyxy": box.tolist(),
                "confidence": float(conf)
            })

    all_frames.append(frame_data)

with open(os.path.join(OUTPUT_DIR, OUTPUT_JSON), "w") as f:
    json.dump(all_frames, f, indent=2)

print("Detections exported to JSON.")