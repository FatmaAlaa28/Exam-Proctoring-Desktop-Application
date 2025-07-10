import cv2
from ultralytics import YOLO
import numpy as np
import pickle
from finalFaceTrain import get_embedding, recognize_face  # Import face recognition models
from mtcnn import MTCNN  # MTCNN for face detection

detector_face = MTCNN()  # Face detection for recognition

# Load face recognition database
with open('finalFace.pkl', 'rb') as f:
    database = pickle.load(f)


# Load a model
model = YOLO(r"Trained Models\yolo11n-pose.pt")

# cap = cv2.VideoCapture('cc.mp4')
cap = cv2.VideoCapture(0)
# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
output_width = 640
output_height = 480

out = cv2.VideoWriter(r'StartVersion (first term)\Recorded Posees\posees.mp4', fourcc, 30, (output_width, output_height))

# Define a function to calculate the angle between three keypoints
def calculate_angle(a, b, c):
    a = np.array(a.cpu())  # First, move tensor to CPU and convert to numpy array
    b = np.array(b.cpu())  # Mid
    c = np.array(c.cpu())  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

while True:
    ret, frame = cap.read()

    try:
        frame = cv2.resize(frame, (output_width, output_height))
    except:
        pass

    if not ret:
        break
    
    
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
    

    results = model.predict(frame, save=True)

    # Get the bounding box information in xyxy format
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    statuses = []

    # Get the keypoints data for all detected persons
    keypoints_data = results[0].keypoints.data

    # Iterate through the detected persons
    for i, keypoints in enumerate(keypoints_data):
        # Ensure keypoints are detected
        if keypoints.shape[0] > 0:
            angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])  # Angle between head, hips, and knees
            print(f"Person {i + 1} is {'Sitting' if angle is not None and angle < 140 else 'Standing'} (Angle: {angle:.2f} degrees)")
            statuses.append('Sitting' if angle is not None and angle < 140 else 'Standing')

    # Draw bounding boxes and statuses on the frame using OpenCV
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Define font and scale for text
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        font_thickness = 2

        # Calculate text size to center it inside the rectangle
        (text_width, text_height), _ = cv2.getTextSize(statuses[i], font, font_scale, font_thickness)
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y2 - 10

        # Draw a filled rectangle for text background
        cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 255, 0), -1)

        # Put text on the frame
        cv2.putText(frame, statuses[i], (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Write the frame to the output video file
    out.write(frame)

    # Display the resulting image with pose detection and statuses
    cv2.imshow('YOLOv11 Pose Detection', frame)

    # Exit the program if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device, close the output video file, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
