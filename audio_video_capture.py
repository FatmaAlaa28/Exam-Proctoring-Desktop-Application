import sounddevice as sd
import numpy as np
import threading
import time
import json
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Shared list for audio violations
audio_violations = []
audio_violations_lock = threading.Lock()

# Variable to store the audio stream
audio_stream = None

def log_violation(student_id):
    violation_data = {
        "type": "audio_detected",
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
        logging.debug(f"Logged audio violation for student {student_id}")
    except FileNotFoundError:
        with open("violation.json", "w") as file:
            json.dump([violation_data], file, indent=4)
        logging.debug(f"Created violation.json and logged audio violation for student {student_id}")
    except Exception as e:
        logging.error(f"Error logging audio violation: {str(e)}")

def start_detection(callback, student_id="unknown"):
    global audio_stream
    logging.debug("Audio stream started. Listening for voices...")

    def audio_callback(indata, frames, time_info, status):
        if status:
            logging.error(f"Audio callback error: {status}")
        # Simple voice detection logic
        amplitude = np.abs(indata).mean()
        if amplitude > 0.05:  # Example threshold
            logging.debug("Voice detected")
            # Simulate multiple voices detection (placeholder)
            num_voices = 2  # Replace with actual voice counting logic
            if num_voices > 1:
                logging.debug("Multiple voices detected!")
                violation = {
                    "student_id": student_id,
                    "cheating_type": "audio_detected",
                    "confidence": 0.9,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with audio_violations_lock:
                    audio_violations.append(violation)
                log_violation(student_id)
                callback(violation)

    # Start the audio stream
    audio_stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=44100,
        blocksize=2048
    )
    audio_stream.start()

def stop_audio_stream():
    global audio_stream
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
        logging.debug("Audio stream stopped")

def main():
    def dummy_callback(violation):
        logging.debug(f"Audio violation detected: {violation}")

    start_detection(dummy_callback, "test_student")
    logging.info("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_audio_stream()

if __name__ == "__main__":
    main()