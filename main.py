import sys
import logging
import mysql.connector
from PyQt5.QtWidgets import QApplication
from sort.sort import Sort
from ultralytics import YOLO
from util import get_car, read_license_plate
from ui import VideoApp
import base64
import cv2
import requests  # Для отправки HTTP-запросов

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models
try:
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    sys.exit(1)

mot_tracker = Sort()
vehicles = [2, 3, 5, 7]  # IDs of vehicles in COCO

# Database configuration
db_config = {
    'host': '192.168.1.159',
    'port': 3306,
    'user': 'iwillnvrd',
    'password': 'SecurePass1212_',
    'database': 'mydatabase'
}


def send_frame(self, frame):
    try:
        if not self.server_ip or not self.server_port:
            logging.error("Server IP or port not set")
            return

        small_frame = cv2.resize(frame, (1280, 720))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', small_frame, encode_param)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        response = requests.post(
            f"http://{self.server_ip}:{self.server_port}/upload",
            json={"frame": jpg_as_text},
            timeout=0.5
        )

        if response.status_code != 200:
            logging.error(f"Server error: {response.status_code} - {response.text}")
            self.connection_status_changed.emit("disconnected")
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {e}")
        self.connection_status_changed.emit("disconnected")
    except Exception as e:
        logging.error(f"Error in send_frame: {e}")
        self.connection_status_changed.emit("disconnected")

def insert_car_data(license_plate_text, photo, car_type, date):
    try:
        conn = mysql.connector.connect(**db_config)
        logging.info("Successfully connected to the database.")
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO car (photo, car_type, car_number, date)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (photo, car_type, license_plate_text, date))
        conn.commit()
        logging.info(f"Inserted data for license plate: {license_plate_text}")
    except mysql.connector.Error as err:
        logging.error(f"Error inserting data into database: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp(coco_model, license_plate_detector, mot_tracker, vehicles, get_car, read_license_plate,
                      insert_car_data)
    window.show()
    logging.info("Application started.")
    sys.exit(app.exec_())