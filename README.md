# Exam Proctoring Desktop Application

## Overview

This repository contains the desktop application for the AI-Based Exam Proctoring System, designed to monitor online examinations in real-time with advanced anti-cheating measures. The desktop application is built using Python and leverages AI technologies for face recognition, gaze tracking, head movement detection, object detection, and audio monitoring to ensure exam integrity. It communicates with an admin side (hosted in a separate repository) to report cheating incidents.

The admin side, which manages exams, users, and cheating detection reports, is implemented using .NET and SQL Server and is hosted in a separate repository: [AI-Based Exam Proctoring System (Admin Side)](https://github.com/FatmaAlaa28/AI-Based-Exam-Proctoring-System).

## Features

- Real-Time Video Monitoring: Captures and processes webcam video to detect cheating behaviors.
- Face Recognition: Identifies the exam taker using MTCNN and a pre-trained face recognition model.
- Gaze Tracking: Detects eye movements to identify if the examinee is looking away from the screen.
- Head Movement Detection: Monitors head orientation to detect suspicious movements.
- Object Detection: Identifies prohibited objects (e.g., phones, books) in the exam environment.
- Audio Monitoring: Detects unauthorized sounds or voices during the exam.
- Secure Browser Integration: Ensures the exam runs within Safe Exam Browser (SEB) for a controlled environment.
- User Interface: Built with PyQt5, providing a clean interface for login, exam selection, and real-time monitoring.
- Cheating Data Reporting: Sends detected violations to the admin side via RESTful APIs.

## Technology Stack

- Programming Language: Python
- Libraries:
  - OpenCV: For video processing and object detection.
  - Dlib: For face detection and landmark extraction.
  - MTCNN: For advanced face detection.
  - TensorFlow: For machine learning models and face recognition embeddings.
  - PyQt5: For the graphical user interface.
  - PyAudio: For audio capture and monitoring.
  - Aiohttp: For asynchronous HTTP requests to the admin API.
- AI Models: Pre-trained face recognition model (finalFace.pkl) for identifying examinees.
- APIs: Communicates with the admin side's RESTful APIs for authentication and reporting.

## Prerequisites

To set up and run the desktop application, ensure you have the following installed:

- Python (version 3.8 or later)
- Pip (Python package manager)
- Safe Exam Browser (SEB) (required for secure exam environments)
- Webcam and Microphone (for video and audio monitoring)
- Dependencies (listed in requirements.txt):
  - opencv-python
  - dlib
  - mtcnn
  - tensorflow
  - pyqt5
  - pyaudio
  - aiohttp
  - psutil
  - (Add any additional dependencies as needed)

## Installation

1. Clone the Repository:

    git clone [https://github.com/FatmaAlaa28/Exam-Proctoring-Desktop-Application]
    cd <desktop-repository-name>
    
2. Install Dependencies:

- Create a virtual environment (optional but recommended):

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
  
- Install required packages:

    pip install -r requirements.txt
  
3. Set Up the Face Recognition Model:

- Ensure the finalFace.pkl model file is placed in the models directory relative to the main script.
  
- If the model is missing, train or obtain it as per the finalFaceTrain.py script instructions.
  
4. Run the Application:

- Execute the main script:

    python main.py
  
- The application will launch a login window where you can enter your email, password, and server URL.

## Configuration
- Server URL: Update the server URL in the application to point to the admin side's API (e.g., http://localhost:5000).
  
- Model Path: Ensure the finalFace.pkl file is correctly placed in the models directory.
  
- Camera and Microphone: Grant necessary permissions for webcam and microphone access.

## Admin Side Integration
The desktop application communicates with the admin side for authentication and cheating data reporting. To integrate:

1. Ensure the admin side is running and accessible (see the admin repository for setup instructions).
   
2. Configure the desktop application to send cheating detection data to the admin API endpoints:
   
  - POST /api/auth/login: For user authentication and retrieving exam links.
    
  - POST /api/detection/cheating: For sending cheating detection data (e.g., unauthorized person, gaze direction, objects).
    
3. Update the server URL in the login window to match the admin side's API base URL.
   
Admin Repository: AI-Based Exam Proctoring System (Admin Side)[https://github.com/FatmaAlaa28/AI-Based-Exam-Proctoring-System].

## Usage
1. Login: Enter your email, password, and the admin server URL in the login window.
  
2. Select Exam: Choose an exam from the list of available links provided by the admin server.
  
3. Start Monitoring: The application will open the exam in a web view, start video/audio monitoring, and display the webcam feed with detection overlays.
  
4. Cheating Detection: The system monitors for:
   
    - Unauthorized persons (via face recognition).
      
    - Suspicious gaze or head movements.
      
    - Prohibited objects in the environment.
      
    - Unauthorized audio activity.
      
5. End Exam: Click the "End Exam" button to stop monitoring and return to the login screen.
   
Detected violations are saved locally in violation.json and sent to the admin server for review.

## Contributing
Contributions are welcome! To contribute:

(1) Fork the repository.

(2) Create a new branch:

      git checkout -b feature/your-feature

(3) Make your changes and commit:

      git commit -m "Add your feature"

(4) Push to the branch:

      git push origin feature/your-feature

(5) Open a pull request.


## License
This project is licensed under the MIT License. See the [License](/LICENSE) file for details.
# Contact
For questions or support, contact [https://github.com/FatmaAlaa28] or open an issue on this repository.
