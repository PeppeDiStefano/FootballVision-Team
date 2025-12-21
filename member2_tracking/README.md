# Football Vision - Member 2: Player Tracking

This module implements the **Multi-Object Tracking (MOT)** subsystem for the Football Vision project. It receives object detections (bounding boxes) and assigns unique IDs to players across video frames, handling occlusions and re-identification using the **SORT (Simple Online and Realtime Tracking)** algorithm.

## Core Logic & Algorithm
The tracking engine is based on **SORT (Simple Online and Realtime Tracking)**.
* **Implementation:** The core algorithm is located in the `libs/` folder and is sourced from the official repository [abewley/sort](https://github.com/abewley/sort).
* **Integration:** The `sort_tracking.py` script wraps this algorithm to process YOLO detections (JSON) and output consistent trajectory data.

## Validation & Performance
To strictly evaluate performance, the system was tested on the **SoccerNet Tracking 2023** dataset (`snmot-060`), chosen for its high-quality **Ground Truth annotations**.

### Parameter Tuning & Optimization
We conducted multiple experiments to find the optimal balance between tracking stability (IDF1) and detection coverage (MOTA/Recall).

| Experiment | Configuration | MOTA | IDF1 | Recall | ID Sw. | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Test 1** (Baseline) | `max_age=1`, `min_hits=5`, `iou=0.01` | 95.9% | 79.5% | 96.5% | **58** | Good baseline, but `max_age` too low for occlusions. |
| **Test 2** (Strict) | `max_age=50`, `min_hits=3`, `iou=0.3` | 93.8% | 73.1% | 94.4% | 88 | `iou` too strict for fast sprints, causing track fragmentation. |
| **Test 3** (Optimal) | `max_age=25`, `min_hits=3`, `iou=0.05` | **96.8%** | **80.4%** | **97.3%** | 60 | **Best Result.** Balanced buffer and tolerance for fast movements. |

### Final Results Analysis
The **Test 3** configuration was selected for the final pipeline.
* **MOTA (96.8%)**: By lowering the IOU threshold to `0.05`, the system successfully tracks players even during sudden accelerations where bounding box overlap is minimal.
* **IDF1 (80.4%)**: Increasing `max_age` to 25 frames (1 second) allows the tracker to "remember" players during short occlusions, significantly improving ID stability compared to the baseline.

## Video Demo
Here is a demonstration of the optimal configuration (Test 3) on the validation sequence.

https://github.com/user-attachments/assets/8f8e1563-e1f4-4764-9916-2ce7e9fca493