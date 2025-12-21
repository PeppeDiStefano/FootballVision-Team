import json
import numpy as np
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
libs_path = os.path.join(project_root, 'libs')

if libs_path not in sys.path:
    sys.path.append(libs_path)

try:
    from sort import Sort
except ImportError as e:
    print(f"Error to load libs from {libs_path}")
    print(f"Detail: {e}")
    sys.exit(1)

DATA_DIR = os.path.join(project_root, 'data')
INPUT_JSON_PATH = os.path.join(DATA_DIR, 'soccerData_input.json') 
OUTPUT_JSON_PATH = os.path.join(DATA_DIR, 'soccerData_output.json')

# max_age=15, min_hits=3, iou_threshold=0.3
TRACKER_PLAYERS = Sort(max_age=25, min_hits=3, iou_threshold=0.05)

def run_tracker():
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"ERROR: File not found: {INPUT_JSON_PATH}")
        print("Make sure the .json file is placed in the 'data' folder")
        return

    print(f"Reading file: {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, 'r') as f:
        frames_data = json.load(f)

    frames_data.sort(key=lambda x: x['frame_id'])
    
    final_output_data = []

    print("Starting processing...")
    for frame in frames_data:
        frame_id = frame['frame_id']
        dets_list = []
        
        for obj in frame['objects']:
            if obj['class_name'] == 'person':
                bbox = obj['bbox_xyxy']
                conf = obj['confidence']
                dets_list.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])

        if len(dets_list) > 0:
            dets_array = np.array(dets_list)
        else:
            dets_array = np.empty((0, 5))

        track_bbs_ids = TRACKER_PLAYERS.update(dets_array)

        frame_result = {
            "frame_id": frame_id,
            "tracked_objects": []
        }

        for trk in track_bbs_ids:
            track_id = int(trk[4])
            bbox_tracked = [float(trk[0]), float(trk[1]), float(trk[2]), float(trk[3])]
            
            frame_result["tracked_objects"].append({
                "track_id": track_id,
                "class_name": "person",
                "bbox_xyxy": bbox_tracked
            })
        
        final_output_data.append(frame_result)
        
        if frame_id % 100 == 0:
            print(f"Frame {frame_id}: OK ({len(track_bbs_ids)} active)")

    print(f"Saving to: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_output_data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    run_tracker()
