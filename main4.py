import cv2
import dlib
import json
import os
import threading
import time
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from head import detect_head  # Head detection model
from eye import gazeDetection  # Gaze detection model
from object_detection_model import detect_objects  # Object detection model
from audio_video_capture import start_detection, stop_audio_stream, record_video_flag
from finalFaceTrain import get_embedding, recognize_face  # Import face recognition models
from mtcnn import MTCNN  # MTCNN for face detection

# Initialize models and resources
detector_eye = dlib.get_frontal_face_detector()  # Eye detection
detector_face = MTCNN()  # Face detection for recognition

# Load face recognition database
file_path = os.path.join('models', 'finalFace.pkl')
with open(file_path, 'rb') as f:
    database = pickle.load(f)


cam = cv2.VideoCapture(0)  # Webcam capture

# Video settings
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
frame_rate = int(cam.get(cv2.CAP_PROP_FPS)) or 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Output folder for recorded videos
output_folder = "Recorded_Videos"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Global video recording variables
video_writer = None
recording_start_time = None

# Functions to manage video recording
def start_video_recording():
    global video_writer, recording_start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(output_folder, f"two_voices_{timestamp}.avi")
    video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, (frame_width, frame_height))
    recording_start_time = datetime.now()
    print(f"[INFO] Recording started: {video_filename}")

def stop_video_recording():
    global video_writer, recording_start_time
    if video_writer:
        video_writer.release()
        video_writer = None
        recording_start_time = None
        print("[INFO] Recording stopped.")

def run_detection(task, *args):
    start_time = time.time()  # Start time for execution measurement
    thread_name = threading.current_thread().name  # Get thread name
    print(f"[THREAD] Running {task.__name__} on {thread_name}...")

    result = task(*args)  # Run the actual detection with the passed arguments

    elapsed_time = time.time() - start_time  # Measure execution time
    print(f"[THREAD] {task.__name__} finished on {thread_name} in {elapsed_time:.2f} seconds.")
    
    return result


# Start audio detection in a separate thread
print("[INFO] Starting audio detection...")
start_detection()

try:
    with ThreadPoolExecutor(max_workers=5) as executor:  # Use 5 threads for 5 detection models
        print("[INFO] ThreadPoolExecutor started with 5 workers.")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("[ERROR] Could not read frame.")
                break
            
            
            ####face####
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector_face.detect_faces(frame_rgb)

            for result in results:
                 x, y, width, height = result['box']
                 x, y = max(0, x), max(0, y)
                 face = frame_rgb[y:y+height, x:x+width]
                 face = cv2.resize(face, (160, 160))  # Resize as done in training
                 embedding = get_embedding(face)  # Use the original face without augmentation
                 identified = recognize_face(embedding, database)

                    # Draw bounding box and label
                 label = f"{identified['name']} ({identified['label']})"
                 cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                 cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ###########
            faces_eye = detector_eye(frame)  # Detect faces for gaze detection
            futures = {
                executor.submit(run_detection, detect_head, frame): "Head",
                executor.submit(run_detection, gazeDetection, faces_eye, frame): "Gaze",  # Pass both faces_eye and frame
                executor.submit(run_detection, detect_objects, frame): "Objects",
            }
            print(f"[INFO] Submitted {len(futures)} tasks to threads.")

            results = {}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                    print(f"[SUCCESS] {task_name} detection completed.")
                except Exception as e:
                    print(f"[ERROR] Error in {task_name} detection: {e}")

            # Process detection results
            # if "Head" in results:
            #     processed_frame = results["Head"]
            if "Head" in results and results["Head"] is not None:
                processed_frame = results["Head"]
            else:
                processed_frame = frame  # fallback to original frame
            if "Gaze" in results:
                gaze_direction = results["Gaze"]  # Assuming gaze direction is the result from gazeDetection

            if "Objects" in results:
                detections = results["Objects"]

                # Save detections to JSON
                if detections:
                    try:
                        with open("violation.json", "r+") as file:
                            data = json.load(file)
                            if isinstance(data, list):
                                data.extend(detections)
                            else:
                                data = detections
                            file.seek(0)
                            json.dump(data, file, indent=4)
                            file.truncate()
                        print(f"[INFO] Detected objects saved to JSON.")
                    except json.JSONDecodeError:
                        with open("violation.json", "w") as file:
                            json.dump(detections, file, indent=4)
                        print("[WARNING] JSON file was empty or corrupted, reinitialized.")

            # Handle video recording
            if record_video_flag.is_set():
                if video_writer is None:
                    start_video_recording()
                elif (datetime.now() - recording_start_time).total_seconds() >= 10:
                    stop_video_recording()
                    record_video_flag.clear()
                else:
                    video_writer.write(frame)
                    print("[INFO] Writing frame to video.")
            elif video_writer is not None:
                stop_video_recording()

            # Show the processed video stream
            cv2.imshow('Processed Video Stream', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting application...")
                break
finally:
    print("[INFO] Stopping audio detection...")
    stop_audio_stream()
    if video_writer is not None:
        stop_video_recording()
    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Cleanup completed.")
