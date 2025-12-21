import numpy as np
from .sort import Sort
from .bytetrack_lib.byte_tracker import BYTETracker

class TrackerManager:
    def __init__(self, algorithm='sort', config=None):
        self.algorithm = algorithm
        if config is None:
            config = {}

        if self.algorithm == 'sort':
            max_age = config.get('max_age', 25)
            min_hits = config.get('min_hits', 3)
            iou_threshold = config.get('iou_threshold', 0.05)
            
            self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
            print(f"Tracker Activated: SORT (Max Age: {max_age}, IOU Threshold: {iou_threshold})")

        elif self.algorithm == 'bytetrack':
            class ByteArgs:
                def __init__(self):
                    self.track_thresh = 0.5
                    self.track_buffer = 30
                    self.match_thresh = 0.8
                    self.mot20 = False

            args = ByteArgs()
            
            if 'max_age' in config:
                args.track_buffer = config['max_age']
            if 'min_conf' in config:
                args.track_thresh = config['min_conf']
            if 'match_thresh' in config:  
                args.match_thresh = config['match_thresh']

            self.tracker = BYTETracker(args)
            print(f"Tracker Activated: BYTETRACK (Buffer: {args.track_buffer}, Match Threshold: {args.match_thresh})")
        
        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Please use 'sort' or 'bytetrack'.")

    def update(self, detections_list):
        if len(detections_list) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array(detections_list)

        final_tracks = []

        if self.algorithm == 'sort':
            trackers = self.tracker.update(dets)
            
            for trk in trackers:
                x1, y1, x2, y2, trk_id = trk
                final_tracks.append([int(trk_id), x1, y1, x2, y2])

        elif self.algorithm == 'bytetrack':
            img_info = (1080, 1920) 
            img_size = (1080, 1920)
            
            online_targets = self.tracker.update(dets, img_info, img_size)

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                
                x1, y1, w, h = tlwh
                x2 = x1 + w
                y2 = y1 + h
                
                final_tracks.append([int(tid), x1, y1, x2, y2])

        return final_tracks