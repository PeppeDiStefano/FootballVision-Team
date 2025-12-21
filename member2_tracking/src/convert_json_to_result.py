import json
import os

input_json_path = "./data/soccerData_output.json"

output_txt_path = "./data/soccerData_results.txt"

def convert_results_to_txt():
    print(f"Converting {input_json_path} -> {output_txt_path}")

    if not os.path.exists(input_json_path):
        print(f"ERROR: Output file not found: {input_json_path}")
        return

    with open(input_json_path, 'r') as f:
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
         
            line = f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            lines_to_write.append(line)

    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(lines_to_write))

    print(f"Conversion completed, file: {output_txt_path}")

if __name__ == "__main__":
    convert_results_to_txt()