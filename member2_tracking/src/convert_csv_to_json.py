import json
import os
import sys
import argparse

def convert_det_to_json():
    parser = argparse.ArgumentParser(description="Convert MOT detection TXT to JSON")
    
    parser.add_argument('--input', type=str, required=True, help="Path to input detection file (det.txt)")
    parser.add_argument('--output', type=str, default=None, help="Path to output JSON file (optional)")

    args = parser.parse_args()

    input_path = args.input

    if args.output:
        output_path = args.output
    else:
        dir_name = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(dir_name, f"{base_name}.json")

    print(f"Converting: {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    frames_dict = {}

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',')
            
            try:
                frame_id = int(parts[0])
                
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
            
                obj = {
                    "class_id": 0,               
                    "class_name": "person",      
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": conf
                }

                if frame_id not in frames_dict:
                    frames_dict[frame_id] = []
                
                frames_dict[frame_id].append(obj)
            except ValueError:
                continue

    json_output = []
    
    sorted_frame_ids = sorted(frames_dict.keys())
    
    for fid in sorted_frame_ids:
        frame_entry = {
            "frame_id": fid,  
            "objects": frames_dict[fid]
        }
        json_output.append(frame_entry)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"Conversion completed successfully. Saved to: {output_path}")

if __name__ == "__main__":
    convert_det_to_json()