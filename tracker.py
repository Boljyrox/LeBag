import cv2
import threading
import time
import requests
import numpy as np
import json
import os
import fcntl

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

# --------------------------
# State Management
# --------------------------
id_to_label = {}
id_to_match_info = {}
id_to_db_key = {}

known_bag_gallery = {} # db_key -> np.array mapping
db_key_to_label = {}   # db_key -> owner string mapping

DB_FILE = "bag_database.json"

def load_database_safe():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            db = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        return db
    except Exception as e:
        print(f"⚠️ Error loading database: {e}")
        return {}

def update_database_safe(new_data_dict):
    try:
        with open(DB_FILE, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                content = f.read()
                db = json.loads(content) if content else {}
            except json.JSONDecodeError:
                db = {}
            
            for k, v in new_data_dict.items():
                if k not in db:
                    db[k] = v
                else:
                    db[k].update(v)
                    
            f.seek(0)
            f.truncate()
            json.dump(db, f, indent=4)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"⚠️ Error updating database: {e}")

def sync_gallery():
    db = load_database_safe()
    for db_key, info in db.items():
        if "embedding" in info:
            known_bag_gallery[db_key] = np.array(info["embedding"], dtype=np.float32)
            db_key_to_label[db_key] = info.get("owner", "Unknown")

# --------------------------
# Background Video Streamer
# --------------------------
class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"Warning: Could not open video source {src}")
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                if grabbed and frame is not None:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.grabbed, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.stream.release()

# --------------------------
# Re-Identification Extractor
# --------------------------
class ReIDExtractor:
    def __init__(self):
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Loading MobileNetV2 Re-ID Model on {self.device}...")
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        base_model = models.mobilenet_v2(weights=weights)
        self.extractor = base_model.features
        self.extractor.add_module('pooling', nn.AdaptiveAvgPool2d((1, 1)))
        self.extractor.add_module('flatten', nn.Flatten())
        
        self.extractor.to(self.device).eval()
        
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transforms(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.extractor(tensor).cpu().numpy().flatten()
            
        norm = np.linalg.norm(feat)
        if norm > 0:
            return feat / norm
        return feat

def send_to_backend(track_id, label):
    url = "http://localhost:5000/api/luggage"
    payload = {"track_id": track_id, "label": label}
    try:
        print(f"\n[HTTP POST] Sending -> Track ID: {track_id}, Label: '{label}' to {url}")
    except Exception as e:
        print(f"Error sending data to backend: {e}")

def main():
    stream_url = "tcp://192.168.2.2:5000"
    print(f"Connecting to Pi stream at {stream_url}...")
    
    vs = VideoStream(stream_url).start()
    time.sleep(2.0)
    
    print("Loading YOLOv8-nano model...")
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
    
    model = YOLO("yolov8n.pt")
    baggage_classes = [24, 26, 28]

    reid_model = ReIDExtractor()

    # Initial sync
    sync_gallery()
    print(f"Initially loaded {len(known_bag_gallery)} bags with RE-ID embeddings from database.")

    print("Starting tracking loop. Press 'q' to quit.")
    
    frame_count = 0
    
    while True:
        grabbed, frame = vs.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue
            
        frame_count += 1
        
        if frame_count % 30 == 0:
            sync_gallery()
            
        results = model.track(frame, 
                              classes=baggage_classes, 
                              persist=True, 
                              tracker="bytetrack.yaml", 
                              verbose=False)
        
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                if track_id not in id_to_label:
                    feature = reid_model.get_embedding(frame, box)
                    
                    assigned_label = None
                    assigned_db_key = None
                    reid_match_pct = 0.0
                    
                    if feature is not None and len(known_bag_gallery) > 0:
                        best_sim = -1.0
                        best_db_key = None
                        
                        for gal_key, gal_feat in known_bag_gallery.items():
                            sim = np.dot(feature, gal_feat)
                            if sim > best_sim:
                                best_sim = sim
                                best_db_key = gal_key
                                
                        if best_sim >= 0.85:
                            assigned_db_key = best_db_key
                            assigned_label = db_key_to_label.get(best_db_key, "Unknown")
                            reid_match_pct = best_sim * 100.0
                            
                    if assigned_label is not None:
                        id_to_label[track_id] = assigned_label
                        id_to_db_key[track_id] = assigned_db_key
                        id_to_match_info[track_id] = f"{reid_match_pct:.1f}%"
                        print(f"Re-Identified Track ID {track_id} as '{assigned_label}' | Match: {reid_match_pct:.1f}%")
                    else:
                        label = f"Unregistered Bag"
                        id_to_label[track_id] = label
                        id_to_match_info[track_id] = "NEW"
                        send_to_backend(track_id, label)
                        
                elif track_id in id_to_db_key and frame_count % 5 == 0:
                    db_key = id_to_db_key[track_id]
                    if db_key in known_bag_gallery:
                        feature = reid_model.get_embedding(frame, box)
                        if feature is not None:
                            alpha = 0.95
                            merged = alpha * known_bag_gallery[db_key] + (1 - alpha) * feature
                            norm = np.linalg.norm(merged)
                            if norm > 0:
                                updated_feat = merged / norm
                                known_bag_gallery[db_key] = updated_feat
                                def bg_update(key, feat):
                                    update_database_safe({key: {"embedding": feat.tolist()}})
                                t = threading.Thread(target=bg_update, args=(db_key, updated_feat))
                                t.daemon = True
                                t.start()
                
                label_text = id_to_label.get(track_id, "Unknown")
                match_info = id_to_match_info.get(track_id, "")
                cls_name = model.names[cls_id].capitalize()
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 144, 30), 2)
                
                if match_info == "NEW" or match_info == "":
                    display_text = f"ID:{track_id} | {label_text} ({cls_name})"
                    color = (255, 144, 30) # Default Orange
                else:
                    display_text = f"RE-ID: {label_text} | Match: {match_info} ({cls_name})"
                    color = (0, 255, 0) # Green for Re-Identified objects
                    
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(annotated_frame, display_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Luggage Tracking", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
