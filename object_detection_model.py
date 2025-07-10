from ultralytics import YOLO
import cv2
import json
import os
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLO model
model_path = os.path.join(os.path.dirname(__file__), 'Trained Models', 'yolo11n.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"YOLO model file not found at: {model_path}")
model = YOLO(model_path)

def log_violation(objects):
    violation_data = {
        "type": "object_detected",
        "objects": objects,
        "confidence": 0.9,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

    try:
        with open("violation.json", "r+") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
            if not isinstance(data, list):
                data = []
            data.append(violation_data)
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()
        logging.debug(f"Logged object violation: {objects}")
    except FileNotFoundError:
        with open("violation.json", "w") as file:
            json.dump([violation_data], file, indent=4)
        logging.debug(f"Created violation.json and logged object violation: {objects}")
    except Exception as e:
        logging.error(f"Error logging object violation: {str(e)}")

def detect_objects(frame):
    try:
        # Perform object detection
        results = model(frame)
        detected_objects = []

        # Process results, only include 'cell phone'
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                label = model.names[class_id]
                if label == 'cell phone':
                    detected_objects.append(label)
                    logging.debug(f"Detected cell phone with confidence {box.conf[0].item()}")
                else:
                    logging.debug(f"Ignored object: {label}")

        if detected_objects:
            log_violation(detected_objects)
            return {"objects_detected": True, "objects": detected_objects}
        return None
    except Exception as e:
        logging.error(f"Object detection error: {str(e)}")
        return None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open webcam.")
        return

    logging.info("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.error("Cannot read frame from webcam.")
            break

        result = detect_objects(frame)
        if result:
            logging.debug(f"Object detection result: {result}")

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()