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
from pyzbar.pyzbar import decode, ZBarSymbol

# --- CONFIGURATION ---
PI_STREAM_URL = "tcp://192.168.2.2:5000"
SERVER_PORT = 5001

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app)

# Shared State
pending_queue = []
bags = []
bag_id_counter = 1
seen_codes = set()
lock = threading.Lock()

@app.route('/enroll', methods=['POST'])
def enroll_bag_external():
    """Endpoint for the Pi to register a bag directly if needed."""
    data = request.json
    if not data or 'name' not in data:
        return jsonify({"error": "Missing 'name' field"}), 400
    
    with lock:
        pending_queue.append(data['name'])
        print(f"📥 [API] Added '{data['name']}' to pending queue (Total: {len(pending_queue)})")
    return jsonify({"message": "Bag added to queue", "queue_size": len(pending_queue)}), 200

@app.route('/api/pop_pending', methods=['GET'])
def pop_pending():
    """Endpoint for tracker.py to pop the next passenger name."""
    with lock:
        if pending_queue:
            name = pending_queue.pop(0)
            print(f"📤 [API] Popped '{name}' from pending queue (Remaining: {len(pending_queue)})")
            return jsonify({"name": name}), 200
        else:
            return jsonify({"name": None}), 200

@app.route('/api/new_bag', methods=['POST'])
def new_bag():
    """Endpoint to manually add a bag (kept for frontend compatibility)"""
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Optimize buffer for lowest latency
        if not cap.isOpened():
             print("⚠️ Could not open TCP stream. Falling back to macbook webcam (0).")
             cap = cv2.VideoCapture(0)
             current_stream_mode = "LOCAL"
        else:
             current_stream_mode = "TCP"
    except Exception as e:
        print(f"⚠️ Error opening stream: {e}. Falling back to macbook webcam (0).")
        cap = cv2.VideoCapture(0)
        current_stream_mode = "LOCAL"

    last_reconnect_time = time.time()
    reconnect_thread = None
    test_cap_result = []

    consistency_counter = {} # {data: count}
    CONSISTENCY_THRESHOLD = 5 

    print("📷 Scanner Running... Looking for Barcodes & QR Codes (Press 'q' to quit)")

    while True:
        try:
            if current_stream_mode == "LOCAL" and time.time() - last_reconnect_time > 5.0:
                if reconnect_thread is None or not reconnect_thread.is_alive():
                    if not test_cap_result:
                        print(f"🔄 Attempting to reconnect to TCP stream at {PI_STREAM_URL} in background...")
                        def _try_reconnect():
                            temp_cap = cv2.VideoCapture(PI_STREAM_URL)
                            if temp_cap.isOpened():
                                test_cap_result.append(temp_cap)
                            else:
                                temp_cap.release()
                        reconnect_thread = threading.Thread(target=_try_reconnect, daemon=True)
                        reconnect_thread.start()
                        last_reconnect_time = time.time()
            
            if test_cap_result:
                print("✅ Successfully reconnected to TCP stream!")
                cap.release()
                cap = test_cap_result.pop(0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                current_stream_mode = "TCP"

            ret, frame = cap.read()
            if not ret:
                if current_stream_mode == "TCP":
                    print("⚠️ TCP Stream lost! Falling back to macbook webcam.")
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    current_stream_mode = "LOCAL"
                    last_reconnect_time = time.time()
                else:
                    print("Stream lost... waiting for keyframe.")
                    time.sleep(0.1)
                continue

            decoded_objects = decode(frame, symbols=[ZBarSymbol.QRCODE, ZBarSymbol.EAN13])
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
                        
                        if "," in data:
                            parts = data.split(",")
                            if len(parts) >= 3:
                                owner = parts[0].strip()
                                bag_type = parts[1].strip()
                                flight = parts[2].strip()
                        elif ":" in data: # Simple colon format for fallback demo
                             parts = data.split(":")
                             if len(parts) >= 2:
                                 owner = parts[1].strip()
                                 
                        print(f"📥 Scanner mapped '{data}' to Owner: {owner}")
                        
                        with lock:
                            pending_queue.append(owner)
                            print(f"➕ Appended '{owner}' to FIFO queue (Total: {len(pending_queue)})")

                        add_bag(owner, bag_type, flight)
            
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

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    run_scanner()
