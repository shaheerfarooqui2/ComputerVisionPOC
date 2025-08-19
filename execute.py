
import cv2
import numpy as np
import time
# You'll likely need ultralytics for YOLOv8 and potentially other deep learning frameworks
# pip install ultralytics opencv-python

# --- Configuration ---
RTSP_URL = "rtsp://shaheerfarooqui2@gmail.com:@Venger10@192.168.0.100:554/stream1"

# Load your custom-trained YOLOv8 model for detecting trash items and possibly human actions/posture
# You would need to train a model specifically for this, ideally with a dataset
# that includes images/videos of people throwing different types of trash.
# For demonstration, we'll assume you have a 'best.pt' file from training.
# See resources for training custom YOLO models for garbage detection.
YOLO_MODEL_PATH = "path/to/your/custom_yolov8_model/weights/best.pt" # Example path
# Define the classes that your YOLO model was trained to detect
# This should include trash categories (e.g., 'plastic_bottle', 'paper_cup')
# and possibly 'person' if you want to track people and their interaction with trash.
CLASS_NAMES = ["plastic_bottle", "paper_cup", "trash_bag", "person", "dustbin"] # Example classes

# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.5

# Tracking threshold (Intersection Over Union - IoU) for matching objects across frames
# You might need to experiment with this value.
TRACKING_THRESHOLD = 0.7

# --- Initialize YOLO Model (assuming Ultralytics YOLO) ---
try:
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)
except ImportError:
    print("Error: ultralytics library not found. Please install it using 'pip install ultralytics'.")
    exit()
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# --- Initialize DeepSORT Tracker (Simplified) ---
# DeepSORT is a popular multi-object tracker that can be integrated with YOLO.
# For this example, we'll use a very basic tracking approach for conceptual understanding.
# In a real-world scenario, you would integrate a dedicated tracking library like 'sort' or 'deepsort'.
# Example of a simple tracker (for concept only, not a robust solution):
class SimpleTracker:
    def __init__(self):
        self.tracked_objects = {}  # Store objects with unique IDs
        self.next_id = 0

    def update(self, detections):
        updated_detections = []
        # Simulate tracking by assigning new IDs or keeping old ones based on proximity
        for detection in detections:
            # Simplified logic: Assign a new ID for each new detection for now
            # In real DeepSORT, you would compare current detections with existing tracks
            # based on position, size, and potentially appearance features.
            obj_id = self.next_id
            self.next_id += 1
            x1, y1, x2, y2, cls, conf = detection
            updated_detections.append((x1, y1, x2, y2, cls, conf, obj_id))
            self.tracked_objects[obj_id] = detection # Store for later analysis

        # You'd have more sophisticated logic here to handle lost tracks,
        # match new detections to existing tracks, etc.

        return updated_detections

tracker = SimpleTracker()

# --- Main loop for video processing ---
def process_litter_stream():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {RTSP_URL}. Check URL and camera settings.")
        return

    print(f"Connected to RTSP stream: {RTSP_URL}")
    
    # Store history of object positions for action analysis
    # For a real application, consider a more complex data structure (e.g., using DeepSORT output).
    object_history = {} 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from stream. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            time.sleep(5)  # Wait before retrying
            continue

        # --- Object Detection with YOLOv8 ---
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False) # Use verbose=False to suppress output

        detections = [] # Store detections in a format suitable for tracking
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, cls, conf))

        # --- Object Tracking ---
        tracked_detections = tracker.update(detections)

        # --- Littering Detection Logic ---
        persons = []
        trash_items = []
        
        # Separate detected objects into persons and trash categories
        for x1, y1, x2, y2, cls, conf, obj_id in tracked_detections:
            class_name = CLASS_NAMES[cls]
            if class_name == 'person':
                persons.append({'bbox': (x1, y1, x2, y2), 'id': obj_id, 'history': object_history.get(obj_id, [])})
            elif 'trash' in class_name or 'bottle' in class_name or 'cup' in class_name: # Modify based on your trash classes
                trash_items.append({'bbox': (x1, y1, x2, y2), 'id': obj_id, 'history': object_history.get(obj_id, [])})
            
            # Update object history
            if obj_id not in object_history:
                object_history[obj_id] = []
            object_history[obj_id].append({'bbox': (x1, y1, x2, y2), 'frame_num': cap.get(cv2.CAP_PROP_POS_FRAMES)})
            # Limit history length to save memory and only consider recent actions
            object_history[obj_id] = object_history[obj_id][-30:] # Keep last 30 frames

        # Check for littering actions
        for person in persons:
            p_x1, p_y1, p_x2, p_y2 = person['bbox']

            for trash in trash_items:
                t_x1, t_y1, t_x2, t_y2 = trash['bbox']

                # Calculate Intersection Over Union (IoU) between person and trash
                # This helps determine if they are interacting.
                intersection_area = max(0, min(p_x2, t_x2) - max(p_x1, t_x1)) * max(0, min(p_y2, t_y2) - max(p_y1, t_y1))
                person_area = (p_x2 - p_x1) * (p_y2 - p_y1)
                trash_area = (t_x2 - t_x1) * (t_y2 - t_y1)
                union_area = person_area + trash_area - intersection_area
                
                iou = intersection_area / union_area if union_area > 0 else 0

                # Simple littering logic:
                # 1. Trash and person were in close proximity (high IoU).
                # 2. Trash is now moving away from the person.
                # 3. Trash is not near a dustbin (if dustbins are detected).

                if iou > 0.1:  # Person and trash were interacting
                    # Check trash's previous positions
                    if len(trash['history']) > 2:
                        prev_trash_bbox = trash['history'][-2]['bbox']
                        # Check if trash has moved significantly away from the person
                        # You would need a more sophisticated motion detection here.
                        # For example, calculate distance between centroids of person and trash in current vs. previous frames.
                        
                        # Check if trash is moving away from the person
                        # (More robust action recognition models like MoViNet are needed for better accuracy).
                        
                        # Check if trash is not entering a dustbin's bounding box.
                        
                        # If conditions indicate littering:
                        cv2.putText(frame, f"Littering Detected! (Person {person['id']})", 
                                    (p_x1, p_y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (0, 0, 255), 4) # Red bounding box for litterer
                        cv2.rectangle(frame, (t_x1, t_y1), (t_x2, t_y2), (0, 165, 255), 2) # Orange for trash
                        # You might store this event in a database or trigger an alert.

        # --- Draw bounding boxes and labels for all detected objects ---
        for x1, y1, x2, y2, cls, conf, obj_id in tracked_detections:
            label = f"{CLASS_NAMES[cls]}: {conf:.2f} (ID: {obj_id})"
            color = (255, 0, 0) # Default color (Blue)
            if CLASS_NAMES[cls] == 'person':
                color = (0, 255, 0) # Green for persons
            elif 'trash' in CLASS_NAMES[cls] or 'bottle' in CLASS_NAMES[cls] or 'cup' in CLASS_NAMES[cls]:
                color = (0, 165, 255) # Orange for trash

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the frame
        cv2.imshow("Littering Detection", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_litter_stream()
