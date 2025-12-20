# Detection Module – Member A

This folder contains the object detection component of the Football Vision project.
The goal of this module is to detect players and the ball in football broadcast videos,
providing reliable detections for downstream tracking and homography estimation modules.

The detection pipeline is developed in two stages: a baseline evaluation using a pretrained model
and a fine-tuned model trained on a football-specific dataset.

---

## Baseline Detection (Sprint 1)

As an initial step, we evaluated a pretrained YOLOv8n model on football broadcast videos.
The purpose of this baseline was to verify the feasibility of detecting relevant objects
(players and the ball) from a single moving broadcast camera.

The baseline detector was applied directly to video input without any fine-tuning.
Detections are exported frame-by-frame to a structured JSON format, which is later consumed
by the tracking module.

Scripts:
- `inference_yolo.py`: runs pretrained YOLOv8 inference on a video and saves a visualization
- `export_detections.py`: exports raw detections (bounding boxes, classes, confidence) to JSON

---

## Fine-Tuned Detection on Soccana Dataset (Sprint 2)

To improve detection performance in football-specific scenarios, the YOLOv8n model is fine-tuned
on the **Soccana Player Ball Detection v1** dataset.

The dataset provides bounding box annotations for three classes:
- Player
- Ball
- Referee

The dataset is already provided in YOLO format and is therefore used directly without additional
annotation conversion.

The dataset is **not included in this repository** and must be downloaded separately.
The configuration file used for training is provided in:

```

data/soccana_yolo/data.yaml

````

Training is performed by launching YOLOv8 from the `member1_detection` directory to ensure
relative paths and full reproducibility:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data/soccana_yolo/data.yaml epochs=30 imgsz=640 batch=8 name=yolov8_soccana
````

The training process produces fine-tuned model weights (`best.pt`) and standard detection
metrics, including mAP@0.5, which are used for quantitative evaluation in the project report.

---

## Notes

* The Soccana dataset and all training outputs (e.g. images, labels, weights, cache files)
  are intentionally excluded from version control.
* The referee class is included during detection training but is not used in subsequent
  tracking or homography estimation modules.
* This detection module is designed to be modular and independent, providing clean
  interfaces for downstream components.