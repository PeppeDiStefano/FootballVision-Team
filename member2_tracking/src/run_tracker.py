import json
import numpy as np
import os
import sys
import argparse

src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from tracker.manager import TrackerManager
except ImportError as e:
    print(f"Error importing TrackerManager: {e}")
    sys.exit(1)

def run_tracker():
    parser = argparse.ArgumentParser(description="Football Player Tracking CLI")
    
    parser.add_argument('--input', type=str, required=True, help="Path to input JSON file")
    
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'data', 'outputs'),
                        help="Output directory")
    parser.add_argument('--output_name', type=str, default=None, 
                        help="Specific output filename. If empty, automatic naming is used.")

    parser.add_argument('--algo', type=str, default='sort', choices=['sort', 'bytetrack'],
                        help="Algorithm to use")
    
    parser.add_argument('--age', type=int, default=None, 
                        help="Max Age: tracker memory")
    parser.add_argument('--conf', type=float, default=None, 
                        help="Min Conf: minimum confidence (ByteTrack ONLY)")
    
    parser.add_argument('--iou', type=float, default=None, 
                        help="SORT: IOU Threshold | ByteTrack: Match Threshold")
    
    args = parser.parse_args()
    
    if args.algo == 'sort':
        config = {
            'max_age': args.age if args.age is not None else 25,
            'min_hits': 3,
            'iou_threshold': args.iou if args.iou is not None else 0.3
        }
    
    elif args.algo == 'bytetrack':
        config = {
            'max_age': args.age if args.age is not None else 30,
            'min_conf': args.conf if args.conf is not None else 0.3,
            'match_thresh': args.iou if args.iou is not None else 0.95
        }

    print(f"\nSTARTING {args.algo.upper()}")
    print(f"Parameters: {config}")

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.output_name:
        filename = args.output_name
        if not filename.endswith('.json'):
            filename += '.json'
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        filename = f"{base_name}_{args.algo}.json"

    output_path = os.path.join(args.output_dir, filename)

    tracker_manager = TrackerManager(algorithm=args.algo, config=config)

    print(f"Input:  {args.input}")
    with open(args.input, 'r') as f:
        frames_data = json.load(f)
    frames_data.sort(key=lambda x: x['frame_id'])
    
    final_output_data = []
    print(f"Processing {len(frames_data)} frames...")

    for frame in frames_data:
        frame_id = frame['frame_id']
        dets_list = []
        for obj in frame['objects']:
            if obj['class_name'] == 'person':
                bbox = obj['bbox_xyxy']
                conf = obj['confidence']
                dets_list.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])

        dets_array = np.array(dets_list) if len(dets_list) > 0 else np.empty((0, 5))
        current_tracks = tracker_manager.update(dets_array)

        frame_result = {"frame_id": frame_id, "tracked_objects": []}
        for trk in current_tracks:
            track_id = int(trk[0])
            bbox = [float(x) for x in trk[1:5]]
            frame_result["tracked_objects"].append({
                "track_id": track_id,
                "class_name": "person",
                "bbox_xyxy": bbox
            })
        final_output_data.append(frame_result)
        
        if frame_id % 100 == 0:
            print(f"   Frame {frame_id}: {len(current_tracks)} tracks active")

    with open(output_path, 'w') as f:
        json.dump(final_output_data, f, indent=2)
    print(f"Done! Saved to: {output_path}")

if __name__ == "__main__":
    run_tracker()