import motmetrics as mm
import numpy as np
import os
import sys
import argparse

def load_mot_file(filepath):
    """
    Reads a file in MOT Challenge format (CSV).
    Format: frame, id, x, y, w, h, conf, ...
    Returns a dictionary: { frame_id: {'ids': [id1, id2], 'bboxes': [[x,y,w,h], ...]} }
    """
    data = {}
    
    if not os.path.exists(filepath):
        print(f"Error: File not found -> {filepath}")
        return {}
    
    print(f"Loading file: {filepath}...")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split(',')
            if len(parts) < 6:
                parts = line.split()
            
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
    parser = argparse.ArgumentParser(description="Evaluate Tracking Results (MOT Format .txt)")
    
    parser.add_argument('--gt', type=str, required=True, help="Path to Ground Truth file (.txt)")
    parser.add_argument('--pred', type=str, required=True, help="Path to Tracking Result file (.txt)")
    parser.add_argument('--iou', type=float, default=0.5, help="IOU Threshold for matching (default 0.5)")
    
    args = parser.parse_args()

    gt_data = load_mot_file(args.gt)
    ts_data = load_mot_file(args.pred)

    if not gt_data or not ts_data:
        print("Error: Unable to calculate metrics due to missing or empty files.")
        sys.exit(1)

    print(f"Calculating metrics with IOU Threshold: {args.iou}")
    
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(list(set(gt_data.keys()) | set(ts_data.keys())))

    for frame in all_frames:
        gt_ids = gt_data.get(frame, {}).get('ids', [])
        gt_bboxes = gt_data.get(frame, {}).get('bboxes', [])
        
        ts_ids = ts_data.get(frame, {}).get('ids', [])
        ts_bboxes = ts_data.get(frame, {}).get('bboxes', [])

        dists = mm.distances.iou_matrix(gt_bboxes, ts_bboxes, max_iou=1.0 - args.iou)
        
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

    print("\n" + "="*80)
    print("TRACKING VALIDATION RESULTS")
    print("="*80)
    
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
    print("="*80)
    print("Legend:")
    print(" - MOTA: Multi-Object Tracking Accuracy (Higher is better)")
    print(" - IDF1: Identification F1 Score (Higher is better, stability)")
    print(" - ID Sw.: Identity Switches (Lower is better)")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_evaluation()