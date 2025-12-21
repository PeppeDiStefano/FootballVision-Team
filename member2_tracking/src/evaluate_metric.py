import motmetrics as mm
import numpy as np
import os
import sys


BASE_DATA_DIR = "./data"
GT_FILE_PATH = os.path.join(BASE_DATA_DIR, "SNMOT-060/gt", "gt.txt")
TS_FILE_PATH = os.path.join(BASE_DATA_DIR, "soccerData_results.txt")

def load_mot_file(filepath):
    """
    Reads a file in MOT Challenge format (CSV).
    Format: frame, id, x, y, w, h, conf, ...
    Returns a dictionary: { frame_id: {'ids': [id1, id2], 'bboxes': [[x,y,w,h], ...]} }
    """
    data = {}
    
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: File not found -> {filepath}")
        return {}
    
    print(f"Loading file: {filepath}...")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split(',')
            
            try:
                frame = int(parts[0])
                obj_id = int(parts[1])

                bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            except ValueError:
                continue 

            if frame not in data:
                data[frame] = {'ids': [], 'bboxes': []}
            
            data[frame]['ids'].append(obj_id)
            data[frame]['bboxes'].append(bbox)
            
    return data

def run_evaluation():
    gt_data = load_mot_file(GT_FILE_PATH)
    ts_data = load_mot_file(TS_FILE_PATH)

    if not gt_data or not ts_data:
        print("Unable to calculate metrics: missing files.")
        return

    print("Calculating metrics... (Frame-by-Frame Analysis)")

    
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(list(set(gt_data.keys()) | set(ts_data.keys())))

    for frame in all_frames:
        gt_ids = gt_data.get(frame, {}).get('ids', [])
        gt_bboxes = gt_data.get(frame, {}).get('bboxes', [])
        
        ts_ids = ts_data.get(frame, {}).get('ids', [])
        ts_bboxes = ts_data.get(frame, {}).get('bboxes', [])

        dists = mm.distances.iou_matrix(gt_bboxes, ts_bboxes, max_iou=0.5)
        
        acc.update(gt_ids, ts_ids, dists)

    mh = mm.metrics.create()
    
    metrics_to_show = [
        'num_frames', 
        'mota', 
        'idf1', 
        'num_switches',  
        'mostly_tracked', 
        'mostly_lost', 
        'precision', 
        'recall'
    ]
    
    summary = mh.compute(acc, metrics=metrics_to_show, name='Overall Summary')

    print("\n" + "="*60)
    print("             TRACKING VALIDATION RESULTS")
    print("="*60)
    
    
    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap={
            'num_frames': 'Frames',
            'mota': 'MOTA (%)',
            'idf1': 'IDF1 (%)',
            'num_switches': 'ID Sw.', 
            'mostly_tracked': 'MT',
            'mostly_lost': 'ML',
            'precision': 'Prec.',
            'recall': 'Recall'
        }
    )
    print(str_summary)
    print("="*60)
    
    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap={
            'num_frames': 'Frames',
            'mota': 'MOTA (%)',
            'idf1': 'IDF1 (%)',
            'id_switches': 'ID Sw.',
            'mostly_tracked': 'MT',
            'mostly_lost': 'ML',
            'precision': 'Prec.',
            'recall': 'Recall'
        }
    )
    print(str_summary)
    print("="*60)
    print("Legend:")
    print(" - MOTA: Overall accuracy (Higher is better, max 100%)")
    print(" - IDF1: Ability to maintain ID (Higher is better)")
    print(" - ID Sw.: Number of times the tracker confused players (Lower is better)")

if __name__ == "__main__":
    run_evaluation()