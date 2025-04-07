import cv2
import numpy as np
from ultralytics import YOLO
from twilio.rest import Client
from collections import deque
import time
import logging

# Configuration
CONFIG = {
    "model": "yolov8s.pt",  # More accurate than 'nano'
    "confidence": 0.6,  # Higher threshold to reduce false positives
    "max_history": 30,  # 1 sec history at 30 FPS
    "anomaly_threshold": 3,  # Sigma multiplier for anomaly detection
    "cooldown": 300,  # 5 minutes between SMS alerts
    "frame_skip": 3,  # Process every 3rd frame
    "twilio": {
        "sid": "ACf351ce4d430d5f46268d7300558fcd66",
        "token": "9aac455b3a3b4f525340ad30c352eba2",
        "from": "+12394455546",
        "to": "+918619801819"
    }
}

# Initialize components
model = YOLO(CONFIG['model'])
sms_client = Client(CONFIG['twilio']['sid'], CONFIG['twilio']['token'])
cap = cv2.VideoCapture(0)
history = deque(maxlen=CONFIG['max_history'])
alert_state = {"last_sent": 0, "retries": 0}
logger = logging.getLogger('people_counter')


def calculate_anomaly(current_count):
    """Detect statistical anomalies using Z-score"""
    if len(history) < 10:  # Need minimum data points
        return False

    arr = np.array(history)
    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:  # No variation in data
        return current_count > mean

    z_score = (current_count - mean) / std
    return abs(z_score) > CONFIG['anomaly_threshold']


def send_sms_alert(count):
    current_time = time.time()
    backoff = min(2 ** alert_state["retries"], 60)

    if (current_time - alert_state["last_sent"]) > backoff:
        try:
            sms_client.messages.create(
                body=f"ALERT: {count} people detected (Anomaly confirmed)",
                from_=CONFIG['twilio']['from'],
                to=CONFIG['twilio']['to']
            )
            alert_state.update({"last_sent": current_time, "retries": 0})
            logger.info("SMS alert sent successfully")
            return True
        except Exception as e:
            logger.error(f"SMS failed: {str(e)}")
            alert_state["retries"] += 1
    return False


# Main processing loop
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % CONFIG['frame_skip'] != 0:
        continue

    # Enhanced detection with tracking
    results = model.track(
        frame,
        classes=0,
        conf=CONFIG['confidence'],
        persist=True,
        tracker="bytetrack.yaml"
    )

    # Post-processing validation
    valid_boxes = []
    for box in results[0].boxes:
        if box.conf >= CONFIG['confidence'] and box.cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Size validation (ignore small boxes)
            if (x2 - x1) * (y2 - y1) > 1000:
                valid_boxes.append(box)

    current_count = len(valid_boxes)
    history.append(current_count)

    # Anomaly detection
    anomaly_detected = calculate_anomaly(current_count)

    # Alert logic with multiple checks
    if current_count > 1 and anomaly_detected:
        send_sms_alert(current_count)

    # Visual feedback with anomaly status
    cv2.putText(frame, f"People: {current_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Anomaly: {anomaly_detected}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show tracked boxes
    for box in valid_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Optimized People Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

