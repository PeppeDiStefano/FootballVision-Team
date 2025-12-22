# Football Vision - Member 2: Player Tracking

This module implements the **Multi-Object Tracking (MOT)** subsystem for the Football Vision project. It processes object detections to assign unique IDs to players across video frames, ensuring consistent trajectories even during occlusions, camera movements, or fast sprints.

## Core Logic & Algorithms

The tracking engine is designed to be modular, supporting two state-of-the-art algorithms for comparative analysis:

* **SORT (Simple Online and Realtime Tracking):** A lightweight algorithm using **Kalman Filtering** for motion prediction and **IOU** (Intersection Over Union) for data association.
  * *Reference Implementation:* [abewley/sort](https://github.com/abewley/sort)
* **ByteTrack:** A more advanced tracker that utilizes a two-stage matching process to recover low-confidence detections, theoretically robust in crowded environments.
  * *Reference Implementation:* [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)

The core implementation is located in the `src/` folder, with `run_tracker.py` serving as the main entry point to process YOLO detections (JSON) and output trajectory data.

## Validation & Performance

To strictly evaluate performance, the system was benchmarked on a "Triangle of Testing" subset from the **SoccerNet Tracking 2023** dataset, chosen to stress-test specific tracking challenges:

1. **Baseline (`SNMOT-060`):** Standard open play with good lighting.
2. **Static Occlusion (`SNMOT-170`):** Corner kick scenario with dense, stationary crowds.
3. **Dynamic Chaos (`SNMOT-165`):** Goal celebration with erratic movements and total occlusions.

### Benchmark Analysis: SORT vs. ByteTrack

We conducted a comprehensive grid search comparing both algorithms across different `max_age` (memory) and `IOU Threshold` (tolerance) values.

| Scenario | Algorithm | Config (Age / IOU) | MOTA | IDF1 | ID Switches | Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | **SORT** | Age=50, IOU=0.05 | **96.8%** | 80.4% | **60** | Both algorithms perform well, but SORT is slightly more stable with fewer ID switches. |
| **Baseline** | ByteTrack | Age=50, IOU=0.05 | 96.5% | **81.8%** | 72 | ByteTrack shows slightly better identity persistence (IDF1) but higher fragmentation. |
| **Static Crowd** | **SORT** | Age=50, IOU=0.05 | 95.5% | **87.8%** | **64** | **Critical Result:** In dense static crowds, SORT remains stable. |
| **Static Crowd** | ByteTrack | Age=50, IOU=0.05 | **95.7%** | 87.8% | **137** | ByteTrack suffers from **2x more ID Switches** (137 vs 64), likely over-associating low-confidence detections in the crowd. |
| **Dynamic Goal** | **SORT** | Age=50, IOU=0.05 | **97.6%** | **93.4%** | **25** | SORT excels in fast motion, recovering players quickly after the celebration. |
| **Dynamic Goal** | ByteTrack | Age=50, IOU=0.05 | 97.4% | 90.8% | 40 | Good performance but SORT remains superior in identity consistency. |


### Final Results Analysis

The benchmark revealed that **SORT outperforms ByteTrack** for this specific application. This result is largely attributed to the **high quality of the input detections** provided by the SoccerNet dataset.

1. **Detection Quality & Algorithm Choice:** ByteTrack is designed to recover low-confidence detections. However, since the input detections are near-perfect, this advantage is nullified. Instead, ByteTrack's complex association logic introduced instability (more ID switches) in crowded scenes compared to SORT's simpler, high-confidence approach. In scenarios with noisier detections, ByteTrack would likely offer a significant performance boost.
2. **Robustness in Statics:** In the Corner Kick scenario (`SNMOT-170`), SORT demonstrated superior stability (**64 switches** vs. **137** for ByteTrack), avoiding the over-association of static players.
3. **Tolerance for Sprints:** A very low **IOU Threshold (`0.05`)** proved essential to track sudden, non-linear sprints without losing the target, a crucial adjustment for the sparsity of a football pitch.

## Video Demo
Here is a demonstration of the optimal configuration (Test 3) on the validation sequence.

https://github.com/user-attachments/assets/8f8e1563-e1f4-4764-9916-2ce7e9fca493


##  Installation

Follow these steps to set up the development environment.

### 1. Create a Virtual Environment

It is better use a virtual environment to isolate dependencies.

**MacOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies

Once the environment is active, install the required libraries:

```bash
pip install -r requirements.txt
```

## Pipeline Usage

The project is divided into 4 main steps.

### 1. Data Preparation (convert_csv_to_json.py)

The tracker works with JSON files. If your data is in the standard MOTChallenge format (det.txt), you must convert it first.

```bash
python src/convert_csv_to_json.py --input data/path/to/det.txt --output data/input_data.json
```

**Available Flags:**
- `--input`: Path to the original detection file (.txt).
- `--output`: (Optional) Path to the destination JSON file.

### 2. Run Tracker (run_tracker.py)

This is the main script that processes frames and assigns IDs to players.

**SORT Example:**

```bash
python src/run_tracker.py --input data/input_data.json --algo sort
```

**ByteTrack Example:**

```bash
python src/run_tracker.py --input data/input_data.json --algo bytetrack --conf 0.3 --output_name results_bt.json
```

**Available Flags:**
- `--input`: JSON file containing detections (Required).
- `--algo`: Algorithm to use: `sort` or `bytetrack` (Default: `sort`).
- `--output_name`: Name of the output file saved in `data/outputs/` (Optional).
- `--conf`: (ByteTrack Only) Minimum confidence to start a new track (e.g., 0.3 for dark videos, default 0.6).
- `--iou`: Intersection Over Union (IoU) threshold. 
  *Note: The interpretation of this value depends on the chosen algorithm:*
  - **For SORT:** Represents the **Minimum Overlap** required. 
    - *Low value (e.g., 0.05)* = **Aggressive**: Matches boxes even with minimal overlap.
    - *High value (e.g., 0.5)* = **Strict**: Requires significant overlap.
  - **For ByteTrack:** Represents the **Maximum Distance** ($1 - IoU$) allowed.
    - *High value (e.g., 0.95)* = **Aggressive**: Matches boxes even if they are far apart (equivalent to 0.05 overlap).
    - *Low value (e.g., 0.1)* = **Strict**: Matches only if boxes are nearly identical (equivalent to 0.9 overlap).
- `--age`: Tracker memory (number of frames to keep a lost ID alive).

### 3. Video Visualization (visualize_tracking.py)

Generates a video with bounding boxes and IDs overlaid to visually verify the results.

```bash
python src/visualize_tracking.py --video data/original_video.mp4 --json data/outputs/results_bt.json
```

**Available Flags:**
- `--video`: Path to the original video (.mp4) or image folder.
- `--json`: Path to the JSON file generated by the tracker.
- `--output`: (Optional) Path/Name of the final output video.

### 4. Metric Evaluation (Benchmark)

To calculate standard metrics like MOTA and IDF1, you need to compare the output with the Ground Truth (GT).

**Step A: Convert JSON Output to TXT**

```bash
python src/convert_json_to_result.py --input data/outputs/results_bt.json
```

This creates a `.txt` file in the same directory as the JSON.

**Step B: Calculate Metrics**

```bash
python src/evaluate_metric.py --gt data/path/to/gt.txt --pred data/outputs/results_bt.txt
```

**Available Flags:**
- `--gt`: Official Ground Truth file (.txt).
- `--pred`: Converted prediction file (.txt).
- `--iou`: IOU Threshold to consider a prediction correct (Default 0.5).

