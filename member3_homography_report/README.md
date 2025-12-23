# Member 3 — Homography & Top-Down Projection

This module implements the **homography estimation and projection pipeline** for the FootballVision project.  
The goal is to map player detections and tracking results from the broadcast video view to a **top-down soccer pitch representation**.

---

## 📌 Overview

The pipeline follows these steps:

1. Extract a reference frame from the broadcast video
2. Manually select corresponding keypoints on:
   - the broadcast frame
   - the pitch template
3. Compute the homography matrix using **RANSAC**
4. Visually validate the homography
5. Project tracked player positions onto the pitch
6. Generate a top-down visualization video

All steps are **fully reproducible** and consistent with the project report.

---

## 📁 Folder Structure

member3_homography_report/
│
├── data/
│ ├── pitch_template.jpg
│ ├── homography_frame.jpg
│ ├── src_points.npy
│ ├── dst_points.npy
│
├── src/
│ ├── extract_homography_frame.py
│ ├── select_homography_points.py
│ ├── compute_homography.py
│ ├── preview_homography.py
│ ├── map_tracking_to_pitch.py
│
├── outputs/
│ ├── homography_matrix.npy
│ ├── tracked_topdown.mp4
│
├── README.md


---

## ▶️ How to Run the Pipeline

All commands must be executed from the `member3_homography_report` directory.

### 1️⃣ Extract Reference Frame

```bash
python src/extract_homography_frame.py


Extracts a central frame from the broadcast video and saves it as:

data/homography_frame.jpg

2️⃣ Select Homography Points (Manual)
python src/select_homography_points.py


Click corresponding points on the broadcast frame

Press q when done

Click the same points in the same order on the pitch template

Press q to finish

The script saves:

data/src_points.npy

data/dst_points.npy

3️⃣ Compute Homography Matrix
python src/compute_homography.py


Computes the homography using RANSAC

Saves the matrix to:

outputs/homography_matrix.npy

4️⃣ Visual Validation
python src/preview_homography.py


Projects the source points onto the pitch

Red dots: projected points

Green dots: ground truth points

Used for qualitative validation of the homography

5️⃣ Map Tracking Results to Pitch
python src/map_tracking_to_pitch.py


Loads tracking results from member2_tracking

Applies the homography to player bottom-center positions

Generates the final top-down video:

outputs/tracked_topdown.mp4

🧠 Notes

All homography computations are performed in pixel coordinates

No scaling or post-hoc normalization is applied

Validation is performed visually, as required by the project scope

The pipeline is modular and independent from detection/tracking code

✅ Output Example

The final output is a top-down video showing player trajectories projected onto the soccer pitch, with consistent player identities and spatial coherence.
