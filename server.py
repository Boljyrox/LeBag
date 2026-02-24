import cv2
import numpy as np
import time
import threading
import json
import os
import fcntl
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from pyzbar.pyzbar import decode

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

# --- CONFIGURATION ---
PI_STREAM_URL = "tcp://192.168.2.2:5000"
SERVER_PORT = 5001

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app)

# Shared State
bags = []
bag_id_counter = 1
seen_codes = set()
lock = threading.Lock()
bag_db = {} # In-memory cache of the database

DB_FILE = "bag_database.json"

def load_database_safe():
    global bag_db
    if not os.path.exists(DB_FILE):
        print("ℹ️ No database file found (yet).")
        return
    try:
        with open(DB_FILE, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            bag_db = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        print(f"📂 Loaded {len(bag_db)} bags from database.")
    except Exception as e:
        print(f"⚠️ Error loading database: {e}")

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

class ReIDExtractor:
    """
    Lightweight feature extractor utilizing MobileNetV2 pretrained on ImageNet.
    Extracts a high-dimensional visual embedding for cosine similarity matching.
    """
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

@app.route('/api/new_bag', methods=['POST'])
def new_bag():
    """Endpoint to manually add a bag (kept for compatibility/testing)"""
    data = request.json
    if not data or 'owner' not in data or 'type' not in data or 'flight' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    add_bag(data['owner'], data['type'], data['flight'])
    return jsonify({"message": "Bag added successfully"}), 201

@app.route('/api/bags', methods=['GET'])
def get_bags():
    """Endpoint for frontend to poll"""
    with lock:
        return jsonify(bags), 200

def add_bag(owner, bag_type, flight):
    """Helper to safely add a bag to the list"""
    global bag_id_counter
    with lock:
        new_bag_entry = {
            "id": bag_id_counter,
            "owner": owner,
            "type": bag_type,
            "flight": flight,
            "timestamp": datetime.now().isoformat()
        }
        bags.insert(0, new_bag_entry)
        bag_id_counter += 1
        print(f"✅ New Bag Added: {new_bag_entry}")

def run_server():
    """Function to run Flask in a thread"""
    print(f"🚀 LeBag Server Backend Running on http://localhost:{SERVER_PORT}")
    app.run(host='0.0.0.0', port=SERVER_PORT, debug=False, use_reloader=False)

def run_scanner():
    """Main Scanner Loop"""
    global seen_codes
    
    print(f"📷 Scanner Connecting to {PI_STREAM_URL}...")
    try:
        cap = cv2.VideoCapture(PI_STREAM_URL)
        if not cap.isOpened():
             print("⚠️ Could not open stream. Falling back to default camera (0).")
             cap = cv2.VideoCapture(0)
    except Exception as e:
        print(f"⚠️ Error opening stream: {e}. Falling back to default camera (0).")
        cap = cv2.VideoCapture(0)

    print("Loading YOLOv8-nano model and ReID Extractor...")
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
    
    model = YOLO("yolov8n.pt")
    baggage_classes = [24, 26, 28]
    reid_model = ReIDExtractor()

    consistency_counter = {} # {data: count}
    CONSISTENCY_THRESHOLD = 5 

    print("📷 Scanner Running... Looking for Barcodes & QR Codes (Press 'q' to quit)")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Stream lost... waiting for keyframe.")
                time.sleep(0.1)
                continue

            decoded_objects = decode(frame)
            current_frame_codes = set()

            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                code_type = obj.type
                current_frame_codes.add(data)
                
                x, y, w, h = obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                consistency_counter[data] = consistency_counter.get(data, 0) + 1
                
                if consistency_counter[data] >= CONSISTENCY_THRESHOLD:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    label = f"{code_type}: {data}"
                    cv2.putText(frame, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if data not in seen_codes:
                        print(f"✅ CONFIRMED NEW {code_type}: {data}")
                        seen_codes.add(data)
                        
                        owner, bag_type, flight = "Unknown", data, "Unknown"
                        
                        if data in bag_db:
                            info = bag_db[data]
                            owner = info.get('owner', 'Unknown')
                            bag_type = info.get('type', 'Unknown')
                            flight = info.get('flight', 'Unknown')
                            print(f"📂 Found in Database: {owner} - {flight}")
                        elif "," in data:
                            parts = data.split(",")
                            if len(parts) >= 3:
                                owner = parts[0].strip()
                                bag_type = parts[1].strip()
                                flight = parts[2].strip()
                        
                        add_bag(owner, bag_type, flight)

                        # Feature Extraction
                        embedding_list = None
                        results = model(frame, classes=baggage_classes, verbose=False)
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            qr_center_x, qr_center_y = x + w/2, y + h/2
                            best_box = None
                            for box in boxes:
                                bx1, by1, bx2, by2 = box
                                if bx1 <= qr_center_x <= bx2 and by1 <= qr_center_y <= by2:
                                    best_box = box
                                    break
                            if best_box is None:
                                best_box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

                            feature = reid_model.get_embedding(frame, best_box)
                            if feature is not None:
                                embedding_list = feature.tolist()
                                print(f"✨ Extracted visual embedding for {data}")

                        # Update database
                        new_db_entry = {
                            data: {
                                "owner": owner,
                                "type": bag_type,
                                "flight": flight
                            }
                        }
                        if embedding_list:
                            new_db_entry[data]["embedding"] = embedding_list
                            
                        update_database_safe(new_db_entry)
                        bag_db.update(new_db_entry)
            
            for code in list(consistency_counter.keys()):
                if code not in current_frame_codes:
                    consistency_counter[code] = 0
                    
        except Exception as e:
            print(f"Error in scanner loop: {e}")
            pass

        cv2.imshow("LeBag Scanner System", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    load_database_safe()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    run_scanner()
