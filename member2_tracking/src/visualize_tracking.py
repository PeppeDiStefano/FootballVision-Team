import cv2
import os
import numpy as np
import json
import sys


INPUT_SOURCE = "./data/SNMOT-060/img1"

TRACKING_JSON = "./data/soccerData_output.json"

OUTPUT_VIDEO = "video_soccernet.mp4"

# ------------------------------------------

def load_json_data(json_path):
    
    print(f"Read JSON: {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    tracker_map = {}
    count_objs = 0

    for frame_entry in data:
        fid = frame_entry['frame_id']
        fid = int(fid)
        
        if fid not in tracker_map:
            tracker_map[fid] = []
            
        for obj in frame_entry.get('tracked_objects', []):
            track_id = obj['track_id']
            bbox = obj['bbox_xyxy'] 
            
            tracker_map[fid].append({
                'track_id': track_id,
                'bbox': bbox
            })
            count_objs += 1
            
    print(f"   Caricati {count_objs} oggetti tracciati su {len(tracker_map)} frame.")
    return tracker_map

def get_frame_generator(source):
    
    if os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"impossible to open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        def vid_gen():
            idx = 1
            while True:
                ret, frame = cap.read()
                if not ret: break
                yield idx, frame
                idx += 1
            cap.release()
        return vid_gen(), fps, (w, h), total

    elif os.path.isdir(source):
        # CARTELLA IMMAGINI
        images = [x for x in os.listdir(source) if x.lower().endswith(('.jpg', '.png'))]
        images.sort() 
        if not images: raise ValueError("Folder is empty")
        
        first = cv2.imread(os.path.join(source, images[0]))
        h, w, _ = first.shape
        total = len(images)
        fps = 25.0 
        
        def img_gen():
            for i, name in enumerate(images):
                frame = cv2.imread(os.path.join(source, name))
                yield i + 1, frame 
        return img_gen(), fps, (w, h), total
    
    else:
        raise ValueError("Invalid input (neither file nor folder).")

def run_visualizer():
    
    try:
        tracker_data = load_json_data(TRACKING_JSON)
    except Exception as e:
        print(f"Data Error: {e}")
        return


    try:
        frame_gen, fps, size, total_frames = get_frame_generator(INPUT_SOURCE)
    except Exception as e:
        print(f"Video Input Error: {e}")
        return

    print(f"Creating video: {size[0]}x{size[1]} @ {fps:.2f} fps")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, size)

    
    np.random.seed(42)
    colors = np.random.randint(0, 255, (2000, 3)).tolist()

    for frame_idx, frame in frame_gen:
        
       
        if frame_idx in tracker_data:
            objects = tracker_data[frame_idx]
            
            for obj in objects:
                tid = obj['track_id']
                bbox = obj['bbox']
                
              
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[2]), int(bbox[3])
                
                
                color = colors[tid % len(colors)]
                
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                
                label = f"{tid}"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
               
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        out.write(frame)
        
        if frame_idx % 50 == 0:
            print(f"Processing {frame_idx}/{total_frames}")

    out.release()
    print(f"\nVideo completed: {os.path.abspath(OUTPUT_VIDEO)}")

if __name__ == "__main__":
    run_visualizer()