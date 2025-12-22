import json
import os
import sys
import argparse

def convert_results_to_txt():
    parser = argparse.ArgumentParser(description="Convert Tracking JSON results to MOT TXT format")

    parser.add_argument('--input', type=str, required=True, help="Path to input tracking JSON file")
    parser.add_argument('--output', type=str, default=None, help="Path to output TXT file (optional)")

    args = parser.parse_args()
    input_path = args.input

    if args.output:
        output_path = args.output
    else:
        # Auto-generate output filename: input.json -> input.txt
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.txt"

    print(f"Converting: {input_path} -> {output_path}")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'r') as f:
        data = json.load(f)

    lines_to_write = []

    for frame_entry in data:
        frame_id = frame_entry['frame_id']
        
        for obj in frame_entry.get('tracked_objects', []):
            track_id = obj['track_id']
            bbox_xyxy = obj['bbox_xyxy']
            
            x1, y1, x2, y2 = bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]
            
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
         
            # MOT Format: frame, id, x, y, w, h, conf, -1, -1, -1
            line = f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            lines_to_write.append(line)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines_to_write))

    print(f"Conversion completed. Saved to: {output_path}")

if __name__ == "__main__":
    convert_results_to_txt()