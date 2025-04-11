import sys
import logging

import torch
from PyQt5.QtWidgets import QApplication
from sort.sort import Sort
from ultralytics import YOLO
from ui import VideoApp
from util import read_license_plate, get_car, insert_car_data

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load models
        coco_model = YOLO('models/yolo11n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        license_plate_detector = YOLO('models/license_plate_detector.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        sys.exit(1)

    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]  # IDs of vehicles in COCO

    app = QApplication(sys.argv)
    window = VideoApp(coco_model, license_plate_detector, mot_tracker,
                     vehicles, get_car, read_license_plate, insert_car_data)
    window.show()
    logging.info("Application started.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
