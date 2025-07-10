import dlib
import cv2
import numpy as np
import time
import json
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load shape predictor model
shapePredictorModel = os.path.join(os.path.dirname(__file__), 'Trained Models', 'shape_predictor_68_face_landmarks.dat')
if not os.path.exists(shapePredictorModel):
    raise FileNotFoundError(f"Shape predictor file not found at: {shapePredictorModel}")
shapePredictor = dlib.shape_predictor(shapePredictorModel)

# Initialize face detector
faceDetector = dlib.get_frontal_face_detector()

# Global variables
left_counter = 0
right_counter = 0
start_time = 0
current_direction = None
continuous_gaze_threshold = 2  # Threshold in seconds for continuous gaze
right_counter_3 = 0
left_counter_3 = 0
violation_log_file = 'violation.json'

def createMask(frame):
    if frame is None:
        logging.error("Frame is empty.")
        return None

    if len(frame.shape) == 3:  # Color image
        height, width, channels = frame.shape
    elif len(frame.shape) == 2:  # Grayscale image
        height, width = frame.shape
    else:
        raise ValueError("Unexpected frame shape.")

    mask = np.zeros((height, width), np.uint8)
    return mask

def extractEye(mask, region, frame):
    cv2.polylines(mask, region, True, 255, 2)
    cv2.fillPoly(mask, region, 255)
    eyes = cv2.bitwise_and(frame, frame, mask=mask)
    return eyes

def eyeSegmentationAndReturnWhite(img, side):
    height, width = img.shape
    if side == 'left':
        img = img[0:height, 0:int(width/2)]
        return cv2.countNonZero(img)
    else:
        img = img[0:height, int(width/2):width]
        return cv2.countNonZero(img)

def log_violation(direction):
    global left_counter, right_counter
    violation_data = {
        "type": "gaze_direction",
        "direction": direction,
        "confidence": 0.9,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

    try:
        with open(violation_log_file, "r+") as file:
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
        logging.debug(f"Logged gaze violation: {direction}")
    except FileNotFoundError:
        with open(violation_log_file, "w") as file:
            json.dump([violation_data], file, indent=4)
        logging.debug(f"Created violation.json and logged gaze violation: {direction}")
    except Exception as e:
        logging.error(f"Error logging gaze violation: {str(e)}")

def gazeDetection(faces, frame):
    global left_counter, right_counter, right_counter_3, left_counter_3
    global start_time, current_direction

    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 2
    TrialRation = 1.2

    leftEye = [36, 37, 38, 39, 40, 41]
    rightEye = [42, 43, 44, 45, 46, 47]

    result = None

    for face in faces:
        facialLandmarks = shapePredictor(frame, face)
        leftEyeRegion = np.array([(facialLandmarks.part(i).x, facialLandmarks.part(i).y) for i in leftEye], np.int32)
        rightEyeRegion = np.array([(facialLandmarks.part(i).x, facialLandmarks.part(i).y) for i in rightEye], np.int32)

        mask = createMask(frame)
        if mask is None:
            continue
        eyes = extractEye(mask, [leftEyeRegion, rightEyeRegion], frame)

        lmin_x, lmax_x = np.min(leftEyeRegion[:, 0]), np.max(leftEyeRegion[:, 0])
        lmin_y, lmax_y = np.min(leftEyeRegion[:, 1]), np.max(leftEyeRegion[:, 1])
        rmin_x, rmax_x = np.min(rightEyeRegion[:, 0]), np.max(rightEyeRegion[:, 0])
        rmin_y, rmax_y = np.min(rightEyeRegion[:, 1]), np.max(rightEyeRegion[:, 1])

        left = eyes[lmin_y:lmax_y, lmin_x:lmax_x]
        right = eyes[rmin_y:rmax_y, rmin_x:rmax_x]

        if left.size == 0 or right.size == 0:
            logging.warning("Empty eye regions detected.")
            continue

        leftGrayEye = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if len(left.shape) == 3 else left
        rightGrayEye = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if len(right.shape) == 3 else right

        leftTh = cv2.adaptiveThreshold(leftGrayEye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        rightTh = cv2.adaptiveThreshold(rightGrayEye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        leftSideOfLeftEye = eyeSegmentationAndReturnWhite(leftTh, 'right')
        rightSideOfLeftEye = eyeSegmentationAndReturnWhite(leftTh, 'left')
        leftSideOfRightEye = eyeSegmentationAndReturnWhite(rightTh, 'right')
        rightSideOfRightEye = eyeSegmentationAndReturnWhite(rightTh, 'left')

        new_direction = None
        if rightSideOfRightEye >= TrialRation * leftSideOfRightEye:
            new_direction = 'left'
        elif leftSideOfLeftEye >= TrialRation * rightSideOfLeftEye:
            new_direction = 'right'

        if new_direction == current_direction:
            duration = time.time() - start_time
            if duration >= continuous_gaze_threshold:
                if current_direction == 'left':
                    left_counter += 1
                elif current_direction == 'right':
                    right_counter += 1

                if left_counter == 3:
                    log_violation('left')
                    result = {"gaze_detected": True, "direction": "left"}
                    left_counter = 0
                    left_counter_3 += 1
                elif right_counter == 3:
                    log_violation('right')
                    result = {"gaze_detected": True, "direction": "right"}
                    right_counter = 0
                    right_counter_3 += 1

                start_time = time.time()
        else:
            current_direction = new_direction
            start_time = time.time()

    return result

def main():
    global start_time, current_direction
    start_time = time.time()
    current_direction = None

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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(gray_frame)
        result = gazeDetection(faces, frame)
        if result:
            logging.debug(f"Gaze detection result: {result}")

        cv2.imshow("Eye Gaze Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()