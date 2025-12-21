import json
import os


input_csv_path = "./data/SNMOT-060/det/det.txt"
output_json_path = "./data/soccerData_input.json"

def convert_det_to_json():
    print(f"Converting {input_csv_path} -> {output_json_path}")
    
    if not os.path.exists(input_csv_path):
        print(f"ERROR: File not found: {input_csv_path}")
        return

    frames_dict = {}

    with open(input_csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            
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

   
    json_output = []
    
    
    sorted_frame_ids = sorted(frames_dict.keys())
    
    for fid in sorted_frame_ids:
        frame_entry = {
            "frame_id": fid,  
            "objects": frames_dict[fid]
        }
        json_output.append(frame_entry)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"Done, file: {output_json_path}")

if __name__ == "__main__":
    convert_det_to_json()