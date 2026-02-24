import cv2
import json
import os
import sys
from pyzbar.pyzbar import decode

# Helper to input with a prompt
def clean_input(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""

# File path for the database
DB_FILE = "bag_database.json"

def load_database():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_database(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def register_bag():
    db = load_database()
    
    # Initialize Camera
    print("📷 Opening Camera for Registration (Press 'q' to quit)...")
    cap = cv2.VideoCapture(0) # Use default camera
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Show the video feed
        cv2.imshow("Bag Registration Scanner", frame)
        
        # Check for q/Q to quit
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

        # Decode
        decoded_objects = decode(frame)
        for obj in decoded_objects:
            code_data = obj.data.decode('utf-8')
            
            # Pause scanning to get input
            print(f"\n✅ CODE DETECTED: {code_data}")
            
            # Check if already exists
            if code_data in db:
                print(f"⚠️ This code is already registered to: {db[code_data]['owner']}")
                overwrite = clean_input("Overwrite? (y/n): ")
                if overwrite.lower() != 'y':
                    print("Skipping...")
                    continue

            print("--- Enter Passenger Details ---")
            owner = clean_input("Passenger Name: ")
            bag_type = clean_input("Bag Description: ")
            flight = clean_input("Flight Number: ")
            
            if owner and bag_type and flight:
                db[code_data] = {
                    "owner": owner,
                    "type": bag_type,
                    "flight": flight
                }
                save_database(db)
                print(f"🎉 Successfully Registered {owner}'s Bag!")
                
                # Wait a bit to avoid double scanning immediately
                cv2.waitKey(2000)
            else:
                print("❌ Registration Cancelled (Missing fields).")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_bag()
