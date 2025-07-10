import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import time
import pickle
import json
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'Trained Models', 'model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
model = pickle.load(open(model_path, 'rb'))
cols = [pos + dim for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_'] for dim in ('x', 'y')]

# Initialize tracking counters
tracking_data = {
    "Right": {"count": 0, "timestamps": []},
    "Left": {"count": 0, "timestamps": []},
    "Bottom": {"count": 0, "timestamps": []}
}

# Initialize FaceMesh once
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

# Global variable to store last sustained violation
last_violation = None

def extract_features(img):
    try:
        NOSE = 1
        FOREHEAD = 10
        LEFT_EYE = 33
        MOUTH_LEFT = 61
        CHIN = 199
        RIGHT_EYE = 263
        MOUTH_RIGHT = 291

        result = face_mesh.process(img)
        face_features = []

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                        face_features.append(lm.x)
                        face_features.append(lm.y)
        return face_features
    except Exception as e:
        logging.error(f"Error extracting features: {str(e)}")
        return []

def normalize(poses_df):
    try:
        normalized_df = poses_df.copy()
        for dim in ['x', 'y']:
            for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
                normalized_df[feature] = poses_df[feature] - poses_df['nose_'+dim]
            diff = normalized_df['mouth_right_'+dim] - normalized_df['left_eye_'+dim]
            for feature in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim, 'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
                normalized_df[feature] = normalized_df[feature] / diff
        return normalized_df
    except Exception as e:
        logging.error(f"Error normalizing features: {str(e)}")
        return poses_df

def log_violation(direction):
    global last_violation
    violation_data = {
        "type": "head_movement",
        "direction": direction,
        "confidence": 0.9,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    last_violation = {"type": "head_movement", "direction": direction, "confidence": 0.9}

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
        logging.debug(f"Logged head violation: {direction}")
    except FileNotFoundError:
        with open("violation.json", "w") as file:
            json.dump([violation_data], file, indent=4)
        logging.debug(f"Created violation.json and logged head violation: {direction}")
    except Exception as e:
        logging.error(f"Error logging head violation: {str(e)}")

def track_direction(direction):
    global last_violation
    try:
        if direction in tracking_data:
            current_time = time.time()
            timestamps = tracking_data[direction]["timestamps"]

            if not timestamps or timestamps[-1][0] != direction:
                timestamps.append([direction, 1, current_time])
            else:
                timestamps[-1][1] += 1  # Increment the count

            if timestamps[-1][1] >= 10 and current_time - timestamps[-1][2] >= 10:
                log_violation(direction)
                timestamps.clear()
    except Exception as e:
        logging.error(f"Error tracking direction: {str(e)}")
        last_violation = None

def detect_head(input_img):
    global last_violation
    try:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        face_features = extract_features(img)

        if len(face_features):
            face_features_df = pd.DataFrame([face_features], columns=cols)
            face_features_normalized = normalize(face_features_df)
            pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()

            if pitch_pred > 0.3:
                direction = 'Top'
                if yaw_pred > 0.3:
                    direction = 'Top Left'
                elif yaw_pred < -0.3:
                    direction = 'Top Right'
            elif pitch_pred < -0.3:
                direction = 'Bottom'
                if yaw_pred > 0.3:
                    direction = 'Bottom Left'
                elif yaw_pred < -0.3:
                    direction = 'Bottom Right'
            elif yaw_pred > 0.3:
                direction = 'Left'
            elif yaw_pred < -0.3:
                direction = 'Right'
            else:
                direction = 'Forward'

            track_direction(direction)

            if last_violation:
                result = last_violation
                last_violation = None
                logging.debug(f"Detected sustained head movement: {result['direction']}")
                return result

        return None
    except Exception as e:
        logging.error(f"Head detection error: {str(e)}")
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

        result = detect_head(frame)
        if result:
            logging.debug(f"Head detection result: {result}")

        cv2.imshow("Head Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()