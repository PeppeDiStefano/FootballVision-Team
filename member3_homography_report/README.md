# ⚽ Football Vision – Member 3: Homography & Mapping

This module implements the **Homography Transformation and Field Mapping** subsystem for the **Football Vision** project.  
It converts object detections or tracked player coordinates from the broadcast video perspective into a **top-down field view**, enabling spatial reasoning, positional analytics, and team visualization.

---

## Core Logic & Algorithm

The mapping logic is based on **Homography Projection**, a classical computer vision technique for converting points between two planes using a transformation matrix.

### Implementation Overview
- The **homography matrix** is computed from at least 4 manually selected corresponding points between the broadcast frame and the top-view pitch image.  
- Once the matrix `H` is generated, it is applied to every detected or tracked coordinate to obtain its equivalent position on the pitch.  
- The system supports mapping for both **YOLO detections** and **SORT/ByteTrack** tracking outputs.

---

### Modules & Scripts

| Script | Description |
|--------|--------------|
| `extract_frame.py` | Extracts one reference frame from a match video for calibration. |
| `compute_homography_points.py` | Defines correspondences and computes the homography matrix. |
| `save_homography_matrix.py` | Saves the computed `H` matrix as `.npy` for later reuse. |
| `test_homography_mapping.py` | Tests and visualizes the projection of sample points. |
| `map_detections_to_pitch.py` | Maps YOLO detections (JSON) to top-view pitch coordinates. |
| `map_tracking_to_pitch.py` | Maps tracking output (SORT/ByteTrack JSON) onto the pitch. |
| `preview_mapped_detections.py` | Displays a visual preview of all mapped positions. |

---

## Validation & Visualization

The system was validated by comparing projected player positions on the **SoccerNet** broadcast videos against the official **pitch coordinate template**.

- The **red overlay points** show the mapped coordinates of detected/tracked players.  
- Each frame transformation was visually verified to ensure spatial consistency.  
- The homography model was evaluated for alignment accuracy and geometric stability.

---

## Integration with Other Modules

| Input Source | Data Type | Output Target |
|---------------|------------|----------------|
| Member 1 – Detection | `detections.json` | Transformed positions on the pitch |
| Member 2 – Tracking | `tracking_output.json` | Top-down player trajectories |
| Member 3 – Homography | `H.npy`, `pitch_template.jpg` | Unified pitch-mapped visualizations |

The output from this module feeds directly into the **visual analytics** and **possession estimation** pipeline.

---

## Parameter Tuning & Optimization

- Optimal calibration achieved using **8–12 point correspondences**.  
- Homography matrix refined through **least squares error minimization**.  
- Mapping precision was visually validated using **manual point overlay comparison**.

---


## Usage Guide

### 1️⃣ Compute Homography
```bash
python src/compute_homography_points.py

→ Click corresponding points on both the broadcast frame and pitch template.  
→ It will compute and save the transformation matrix as `homography_matrix.npy`.

---

### 2️⃣ Verify Mapping
```bash
python src/test_homography_mapping.py
Displays red markers on the top-down field showing where selected points map.
→ Use this step to confirm alignment visually.

3️⃣ Map YOLO Detections
python src/map_detections_to_pitch.py


→ Loads YOLO detection JSON and projects every bounding box onto the field.
→ Output video: mapped_output.avi.

4️⃣ Map Tracking Results
python src/map_tracking_to_pitch.py


→ Loads tracking JSON (e.g., ByteTrack/SORT) and draws player trajectories.
→ Output video: tracked_topdown.avi.

5️⃣ Preview All Mapped Points
python src/preview_mapped_detections.py

→ Quick visualization to ensure all mappings appear correctly over the pitch image.

Dependencies
Python ≥ 3.9
NumPy
OpenCV
Matplotlib (optional, for visualization)


