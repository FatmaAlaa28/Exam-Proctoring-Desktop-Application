import cv2
import dlib
import json
import os
import threading
import time
import logging
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QSplitter, QComboBox, QMessageBox, QStatusBar)
from PyQt5.QtCore import Qt, QUrl, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush, QLinearGradient, QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView
import aiohttp
import asyncio
import warnings
import tensorflow as tf
import re
import psutil
import pyaudio
import sys
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GLOG_minloglevel"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from mtcnn import MTCNN
from head import detect_head
from eye import gazeDetection
from object_detection_model import detect_objects
from audio_video_capture import start_detection, stop_audio_stream, audio_violations, audio_violations_lock
from finalFaceTrain import get_embedding, recognize_face

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def is_seb_running():
    """Check if Safe Exam Browser is running"""
    for proc in psutil.process_iter(['name']):
        if 'SafeExamBrowser' in proc.info['name']:
            return True
    return False

class ExamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exam Monitoring")
        self.setGeometry(100, 100, 1200, 600)
        self.exam_id = None

        # إعداد خلفية متدرجة
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, 600)
        gradient.setColorAt(0.0, QColor(60, 110, 150))
        gradient.setColorAt(1.0, QColor(150, 200, 255))
        palette.setBrush(QPalette.Background, QBrush(gradient))
        self.setPalette(palette)

        # إعداد asyncio
        self.loop = asyncio.new_event_loop()
        self.session = aiohttp.ClientSession(loop=self.loop)
        self.loop_thread = threading.Thread(target=self.run_loop, daemon=True)
        self.loop_thread.start()
        logging.debug("Asyncio loop thread started")

        # واجهة المستخدم
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # شريط الحالة
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Please enter credentials")

        self.show_login_window()

    def run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def show_login_window(self):
        self.clear_window()
        self.login_widget = QWidget()
        self.login_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        self.login_widget.setFixedSize(700, 500)  # حجم ثابت للتوسيط
        self.login_layout = QVBoxLayout(self.login_widget)
        self.login_layout.setAlignment(Qt.AlignCenter)

        # عنوان
        title_label = QLabel("Exam Monitoring Portal")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2C3E50; margin-bottom: 20px;")
        self.login_layout.addWidget(title_label)

        # حقل البريد الإلكتروني
        email_frame = QHBoxLayout()
        email_label = QLabel("📧")
        email_label.setStyleSheet("font-size: 16px; color: #3498DB; margin-right: 10px;")
        self.email_entry = QLineEdit()
        self.email_entry.setPlaceholderText("Enter your email")
        self.email_entry.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #3498DB;
                border-radius: 5px;
                background-color: #F5F6FA;
            }
        """)
        email_frame.addWidget(email_label)
        email_frame.addWidget(self.email_entry)
        self.login_layout.addLayout(email_frame)

        # حقل كلمة المرور
        password_frame = QHBoxLayout()
        password_label = QLabel("🔒")
        password_label.setStyleSheet("font-size: 16px; color: #3498DB; margin-right: 10px;")
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.password_entry.setPlaceholderText("Enter your password")
        self.password_entry.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #3498DB;
                border-radius: 5px;
                background-color: #F5F6FA;
            }
        """)
        password_frame.addWidget(password_label)
        password_frame.addWidget(self.password_entry)
        self.login_layout.addLayout(password_frame)

        # حقل عنوان الخادم
        server_frame = QHBoxLayout()
        server_label = QLabel("🌐")
        server_label.setStyleSheet("font-size: 16px; color: #3498DB; margin-right: 10px;")
        self.server_entry = QLineEdit()
        self.server_entry.setPlaceholderText("Enter server URL")
        self.server_entry.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #3498DB;
                border-radius: 5px;
                background-color: #F5F6FA;
            }
        """)
        server_frame.addWidget(server_label)
        server_frame.addWidget(self.server_entry)
        self.login_layout.addLayout(server_frame)

        # زر تسجيل الدخول
        self.login_button = QPushButton("Login")
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        self.login_button.clicked.connect(self.handle_login)
        self.login_layout.addWidget(self.login_button)

        # إضافة فواصل للتوسيط العمودي
        self.main_layout.addStretch()
        self.main_layout.addWidget(self.login_widget, alignment=Qt.AlignCenter)
        self.main_layout.addStretch()

    def clear_window(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def validate_server_url(self, url):
        pattern = r'^http(s)?://[\w.-]+(:\d+)?$'
        return bool(re.match(pattern, url))

    async def login_async(self, email, password, server_url):
        try:
            async with self.session.post(
                f"{server_url}/api/auth/login",
                json={"email": email, "password": password},
                timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logging.debug(f"Login response: {data}")
                    links = data.get("links", [])
                    if not links:
                        logging.warning("No links returned from server")
                    return links
                else:
                    response_text = await response.text()
                    logging.error(f"Login failed: {response.status}, {response_text}")
                    return None
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            return None

    def handle_login(self):
        email = self.email_entry.text().strip()
        password = self.password_entry.text().strip()
        server_url = self.server_entry.text().strip()
        if not email or not password or not server_url:
            self.status_bar.showMessage("Please enter email, password, and server URL")
            logging.warning("Email, password, or server URL missing")
            QMessageBox.warning(self, "Warning", "Please enter email, password, and server URL")
            return
        if not self.validate_server_url(server_url):
            self.status_bar.showMessage("Invalid server URL format")
            logging.warning("Invalid server URL format")
            QMessageBox.warning(self, "Warning", "Invalid server URL format")
            return
        self.status_bar.showMessage("Logging in...")
        self.login_button.setEnabled(False)
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.login_async(email, password, server_url),
                self.loop
            )
            links = future.result(timeout=5.0)
            logging.debug(f"Received links: {links}")
            if links is None or not links:
                self.status_bar.showMessage("No exam links available or login failed")
                QMessageBox.critical(self, "Error", "No exam links available or login failed")
                return
            self.status_bar.showMessage("Login successful")
            self.server_url = server_url
            self.show_links_window(links)
        except Exception as e:
            self.status_bar.showMessage(f"Login error - {str(e)}")
            logging.error(f"Login error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Login error: {str(e)}")
        finally:
            self.login_button.setEnabled(True)

    def show_links_window(self, links):
        self.clear_window()
        self.setWindowTitle("Exam Links")

        links_widget = QWidget()
        links_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        links_widget.setFixedSize(700, 500)  # حجم ثابت للتوسيط
        links_layout = QVBoxLayout(links_widget)
        links_layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("Choose Your Exam")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50; margin-bottom: 20px;")
        links_layout.addWidget(title_label)

        self.exam_combo = QComboBox()
        self.exam_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 14px;
                border: 1px solid #3498DB;
                border-radius: 5px;
                background-color: #F5F6FA;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        if not links:
            self.exam_combo.addItem("No exam links available")
            self.exam_combo.setEnabled(False)
        else:
            for link in links:
                self.exam_combo.addItem(link.get("name", "Link"), link)
        links_layout.addWidget(self.exam_combo)

        start_button = QPushButton("Start Exam")
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        start_button.clicked.connect(self.open_exam)
        links_layout.addWidget(start_button)

        # إضافة فواصل للتوسيط العمودي
        self.main_layout.addStretch()
        self.main_layout.addWidget(links_widget, alignment=Qt.AlignCenter)
        self.main_layout.addStretch()
        self.status_bar.showMessage("Select an exam link" if links else "No links available")
        self.initialize_exam_components()

    def initialize_exam_components(self):
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            self.status_bar.showMessage("Error - Camera not found or permission denied")
            logging.error("Could not open camera. Check system permissions or SEB settings")
            raise RuntimeError("Could not open camera")

        try:
            p = pyaudio.PyAudio()
            info = p.get_default_input_device_info()
            if info['maxInputChannels'] == 0:
                raise Exception("No microphone detected")
            p.terminate()
        except Exception as e:
            self.status_bar.showMessage("Error - Microphone not found or permission denied")
            logging.error(f"Could not access microphone: {str(e)}")
            raise RuntimeError("Could not access microphone")

        logging.debug(f"Camera and microphone opened successfully. Resolution: {int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        self.detector_eye = dlib.get_frontal_face_detector()
        self.detector_face = MTCNN()
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'finalFace.pkl')
        if not os.path.exists(model_path):
            self.status_bar.showMessage("Error - finalFace.pkl not found")
            logging.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        with open(model_path, 'rb') as f:
            self.database = pickle.load(f)
        logging.debug("Models initialized successfully")

        self.running = False
        self.executor = None

    def show_monitoring_window(self, exam_url, student_id, exam_id):
        self.clear_window()
        self.setWindowTitle("Exam Monitoring")
        self.exam_id = exam_id
        self.student_id = student_id
        self.exam_url = exam_url

        # إعداد النافذة المقسمة
        splitter = QSplitter(Qt.Horizontal)

        # النصف الأيسر: المراقبة (عرض الفيديو)
        video_widget = QWidget()
        video_widget.setStyleSheet("background-color: #FFFFFF; border-radius: 10px; padding: 10px;")
        video_layout = QVBoxLayout(video_widget)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #3498DB; border-radius: 5px;")
        self.video_label.setFixedSize(840, 680)  # حجم ثابت لشاشة الفيديو
        video_layout.addWidget(self.video_label)

        # زر إنهاء الامتحان
        end_button = QPushButton("End Exam")
        end_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        end_button.clicked.connect(self.stop_exam)
        video_layout.addWidget(end_button)

        splitter.addWidget(video_widget)

        # النصف الأيمن: عرض رابط الامتحان
        web_view = QWebEngineView()
        web_view.setUrl(QUrl(exam_url))
        web_view.setStyleSheet("background-color: #FFFFFF; border-radius: 10px;")
        splitter.addWidget(web_view)

        # ضبط حجم النافذتين
        splitter.setSizes([480, 720])  # 40% فيديو، 60% امتحان
        self.main_layout.addWidget(splitter)

        self.status_bar.showMessage("Monitoring")
        self.start_exam()

    async def send_cheating_data_async(self, cheating_data, server_url):
        retries = 3
        backoff_factor = 1
        for attempt in range(retries):
            try:
                async with self.session.post(
                    f"{server_url}/api/detection/cheating",
                    json=cheating_data,
                    timeout=5
                ) as response:
                    response_text = await response.text()
                    logging.debug(f"Sent cheating data: {cheating_data}, Status: {response.status}, Response: {response_text}")
                    if response.status == 500:
                        logging.warning(f"Server error on attempt {attempt + 1}: {response_text}")
                        if attempt < retries - 1:
                            await asyncio.sleep(backoff_factor * (2 ** attempt))
                            continue
                    self.status_bar.showMessage(f"HTTP {response.status} - {response_text[:50]}")
                    return {"status": response.status, "response": response_text}
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(backoff_factor * (2 ** attempt))
                    continue
                self.status_bar.showMessage(f"Connection Error - {e}")
                return {"status": None, "response": str(e)}
        return {"status": None, "response": "Max retries reached"}

    def send_cheating_data(self, cheating_data, server_url):
        snake_case_data = {
            "student_id": cheating_data["student_id"],
            "exam_id": cheating_data["exam_id"],
            "cheating_type": cheating_data["cheating_type"],
            "confidence": cheating_data["confidence"],
            "timestamp": cheating_data["timestamp"],
            "direction": cheating_data["direction"],
            "detected_id": cheating_data["detected_id"] or "none",
            "objects": cheating_data["objects"],
            "is_seb_running": cheating_data["is_seb_running"]
        }
        logging.debug(f"Preparing to send cheating data: {snake_case_data}")
        future = asyncio.run_coroutine_threadsafe(
            self.send_cheating_data_async(snake_case_data, server_url),
            self.loop
        )
        try:
            result = future.result(timeout=5.0)
            logging.debug(f"Cheating data sent, result: {result}")
            if result["status"] != 200:
                logging.warning(f"Failed to send cheating data: {result}")
        except Exception as e:
            logging.error(f"Error sending cheating data: {str(e)}")

    def start_exam(self):
        if not is_seb_running():
            self.status_bar.showMessage("Error - Safe Exam Browser not running")
            logging.error("Safe Exam Browser is not running")
            QMessageBox.critical(self, "Error", "Safe Exam Browser must be running")
            return
        self.running = True
        try:
            start_detection(lambda x: self.send_cheating_data(x, self.server_url), self.student_id)
            logging.debug("Audio detection started")
        except Exception as e:
            logging.error(f"Failed to start audio detection: {str(e)}")
            self.status_bar.showMessage(f"Audio Detection Error - {e}")
        self.executor = ThreadPoolExecutor(max_workers=3)
        threading.Thread(target=self.video_thread, daemon=True).start()
        logging.debug("Exam started, video thread running")

    def stop_exam(self):
        self.running = False
        stop_audio_stream()
        if self.cam.isOpened():
            self.cam.release()
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        self.status_bar.showMessage("Exam stopped")
        logging.debug("Exam stopped")
        self.show_login_window()

    def video_thread(self):
        frame_count = 0
        while self.running:
            try:
                if not is_seb_running():
                    cheating_data = {
                        "student_id": self.student_id,
                        "exam_id": self.exam_id,
                        "cheating_type": "seb_not_running",
                        "confidence": 1.0,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "direction": "",
                        "detected_id": "none",
                        "objects": [],
                        "is_seb_running": False
                    }
                    self.send_cheating_data(cheating_data, self.server_url)
                    self.status_bar.showMessage("Error - SEB not running")
                    logging.error("Safe Exam Browser stopped")
                    self.stop_exam()
                    break

                ret, frame = self.cam.read()
                frame_count += 1
                if not ret:
                    self.status_bar.showMessage("Camera Error")
                    logging.error(f"Frame {frame_count} - Camera read failed")
                    break

                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not hasattr(self, 'detector_eye') or not hasattr(self, 'detector_face'):
                    logging.error(f"Frame {frame_count} - Detectors not initialized")
                    break

                faces_eye = self.detector_eye(frame)
                face_results = self.detector_face.detect_faces(frame_rgb)
                results = {"cheating": []}

                logging.debug(f"Frame {frame_count} - Eye detector faces: {len(faces_eye)}")
                logging.debug(f"Frame {frame_count} - MTCNN face results: {len(face_results)}")

                for result in face_results:
                    x, y, width, height = result['box']
                    x, y = max(0, x), max(0, y)
                    face = frame_rgb[y:y+height, x:x+width]
                    face = cv2.resize(face, (160, 160))
                    try:
                        embedding = get_embedding(face)
                        identified = recognize_face(embedding, self.database)
                        logging.debug(f"Frame {frame_count} - Face recognition result: {identified}")

                        if identified['name'] != self.student_id and identified['name'] != 'x':
                            violation = {
                                "type": "unauthorized_person",
                                "confidence": 0.95,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "detected_id": identified['name']
                            }
                            results["cheating"].append(violation)
                            with open("violation.json", "r+") as file:
                                try:
                                    data = json.load(file)
                                except json.JSONDecodeError:
                                    data = []
                                if not isinstance(data, list):
                                    data = []
                                data.append(violation)
                                file.seek(0)
                                json.dump(data, file, indent=4)
                                file.truncate()
                        if identified['name'] == 'x':
                            violation = {
                                "type": "unknown_face",
                                "confidence": 0.95,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            results["cheating"].append(violation)
                            with open("violation.json", "r+") as file:
                                try:
                                    data = json.load(file)
                                except json.JSONDecodeError:
                                    data = []
                                if not isinstance(data, list):
                                    data = []
                                data.append(violation)
                                file.seek(0)
                                json.dump(data, file, indent=4)
                                file.truncate()

                        label = f"{identified['name']} ({identified['label']})"
                        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        logging.error(f"Frame {frame_count} - Face recognition error: {str(e)}")

                if self.executor:
                    futures = {
                        self.executor.submit(detect_head, frame): "Head",
                        self.executor.submit(gazeDetection, faces_eye, frame): "Gaze",
                        self.executor.submit(detect_objects, frame): "Objects",
                    }
                    for future in futures:
                        task_name = futures[future]
                        try:
                            result = future.result()
                            logging.debug(f"Frame {frame_count} - {task_name} result: {result}")
                            if task_name == "Head" and result is not None:
                                violation = {
                                    "type": "head_movement",
                                    "confidence": result.get("confidence", 0.9),
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "direction": result.get("direction", "")
                                }
                                results["cheating"].append(violation)
                            elif task_name == "Gaze" and result is not None:
                                violation = {
                                    "type": "gaze_direction",
                                    "confidence": 0.9,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "direction": result.get("direction", "")
                                }
                                results["cheating"].append(violation)
                            elif task_name == "Objects" and result is not None:
                                violation = {
                                    "type": "object_detected",
                                    "confidence": 0.9,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "objects": result.get("objects", [])
                                }
                                results["cheating"].append(violation)
                        except Exception as e:
                            logging.error(f"Frame {frame_count} - Error in {task_name}: {str(e)}")

                with audio_violations_lock:
                    for audio_violation in audio_violations:
                        if audio_violation and isinstance(audio_violation, dict):
                            violation = {
                                "type": audio_violation.get("cheating_type", "audio_detected"),
                                "confidence": audio_violation.get("confidence", 0.9),
                                "timestamp": audio_violation.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                            }
                            results["cheating"].append(violation)
                        else:
                            logging.warning(f"Frame {frame_count} - Invalid audio violation: {audio_violation}")
                    audio_violations.clear()

                for violation in results["cheating"]:
                    if not violation or not isinstance(violation, dict):
                        logging.warning(f"Frame {frame_count} - Skipping invalid violation: {violation}")
                        continue
                    if not violation.get("type") or violation.get("type").startswith("error_"):
                        logging.warning(f"Frame {frame_count} - Skipping error violation: {violation}")
                        continue
                    if "timestamp" not in violation:
                        logging.warning(f"Frame {frame_count} - Missing timestamp in violation: {violation}")
                        violation["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

                    cheating_data = {
                        "student_id": self.student_id,
                        "exam_id": self.exam_id,
                        "cheating_type": violation.get("type", "unknown"),
                        "confidence": float(violation.get("confidence", 0.9)),
                        "timestamp": str(violation.get("timestamp")),
                        "direction": violation.get("direction", ""),
                        "detected_id": violation.get("detected_id", "none"),
                        "objects": violation.get("objects", []),
                        "is_seb_running": is_seb_running()
                    }
                    try:
                        cheating_data_json = json.dumps(cheating_data)
                        logging.debug(f"Frame {frame_count} - Sending raw JSON: {cheating_data_json}")
                        self.send_cheating_data(cheating_data, self.server_url)
                        logging.debug(f"Frame {frame_count} - Reported cheating to server: {cheating_data}")
                    except Exception as e:
                        logging.error(f"Frame {frame_count} - JSON serialization error: {str(e)}")

                try:
                    height, width, channel = frame_rgb.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))  # حجم ثابت للفيديو
                    logging.debug(f"Frame {frame_count} - GUI updated successfully")
                except Exception as e:
                    logging.error(f"Frame {frame_count} - GUI update error: {str(e)}")

                if results["cheating"]:
                    self.status_bar.showMessage(f"Cheating detected - {results['cheating'][0].get('type', 'unknown')}")
                    logging.debug(f"Frame {frame_count} - Cheating detected: {results['cheating']}")
                else:
                    logging.debug(f"Frame {frame_count} - No cheating detected")
                time.sleep(0.033)
            except Exception as e:
                logging.error(f"Frame {frame_count} - Video thread error: {str(e)}")
                break

    def open_exam(self):
        if not is_seb_running():
            self.status_bar.showMessage("Error - Safe Exam Browser not running")
            logging.error("Safe Exam Browser is not running")
            QMessageBox.critical(self, "Error", "Safe Exam Browser must be running")
            return
        selected_link = self.exam_combo.currentData()
        if not selected_link:
            self.status_bar.showMessage("No exam selected")
            logging.error("No exam selected")
            QMessageBox.critical(self, "Error", "Please select an exam")
            return
        url = selected_link.get("url")
        student_id = selected_link.get("student_id")
        exam_id = selected_link.get("exam_id")
        logging.debug(f"Opening exam URL: {url}, Student ID: {student_id}, Exam ID: {exam_id}")
        self.show_monitoring_window(url, student_id, exam_id)

    def closeEvent(self, event):
        self.running = False
        stop_audio_stream()
        if hasattr(self, 'cam') and self.cam.isOpened():
            self.cam.release()
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        if hasattr(self, 'session') and self.session:
            asyncio.run_coroutine_threadsafe(self.session.close(), self.loop)
        if hasattr(self, 'loop'):
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
        logging.debug("Application cleanup completed")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExamApp()
    window.show()
    sys.exit(app.exec_())