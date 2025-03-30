# ui.py
import base64
import logging
import queue
from datetime import datetime
import threading
import time
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QSizePolicy, QTextEdit, QPushButton, QVBoxLayout, QApplication,
    QMenuBar, QMenu, QAction, QInputDialog, QLineEdit, QCheckBox, QDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QFileDialog, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QPen
from PIL import ImageFont, ImageDraw, Image
import cv2
import requests
from util import model_prediction, reader, draw_best_result
from queue import Queue
import socket
import os

from util import draw_license_plate_text

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def get_available_usb_cameras(max_to_check=5):
    available_cameras = []
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

class StatusIndicator(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.set_status("disconnected")

    def set_status(self, status):
        self.status = status
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.status == "connected":
            color = QColor(0, 255, 0)
        elif self.status == "connecting":
            color = QColor(255, 255, 0)
        else:
            color = QColor(255, 0, 0)

        painter.setBrush(color)
        painter.setPen(QPen(Qt.black, 1))
        painter.drawEllipse(2, 2, 16, 16)

        if self.status == "disconnected":
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "!")


class ServerSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки сервера трансляции")
        self.main_layout = QVBoxLayout(self)

        # Form layout for inputs
        self.form_layout = QFormLayout()
        self.main_layout.addLayout(self.form_layout)

        self.ip_input = QLineEdit("192.168.1.159", self)
        self.form_layout.addRow("IP сервера:", self.ip_input)

        self.port_input = QLineEdit("8765", self)
        self.form_layout.addRow("Порт сервера:", self.port_input)

        self.test_button = QPushButton("Проверить подключение", self)
        self.test_button.clicked.connect(self.test_connection)
        self.form_layout.addRow(self.test_button)

        self.status_label = QLabel("Статус: Не проверено", self)
        self.form_layout.addRow(self.status_label)

        # Buttons layout
        self.buttons = QHBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Отмена", self)
        self.cancel_button.clicked.connect(self.reject)
        self.buttons.addWidget(self.ok_button)
        self.buttons.addWidget(self.cancel_button)
        self.main_layout.addLayout(self.buttons)

    def test_connection(self):
        ip = self.ip_input.text()
        port = self.port_input.text()
        try:
            # Test basic connection
            # test_response = requests.get(f"http://{ip}:{port}/test", timeout=2)
            # if test_response.status_code != 200:
            #     self.status_label.setText(f"Статус: Ошибка сервера ({test_response.status_code})")
            #     self.status_label.setStyleSheet("color: red")
            #     return False

            # Test upload endpoint
            upload_response = requests.post(
                f"http://{ip}:{port}/upload",
                json={"frame": "test"},
                timeout=2
            )

            if upload_response.status_code in [200, 405]:  # 405 means endpoint exists
                self.status_label.setText("Статус: Подключение успешно")
                self.status_label.setStyleSheet("color: green")
                return True
            else:
                self.status_label.setText(f"Статус: Ошибка эндпоинта ({upload_response.status_code})")
                self.status_label.setStyleSheet("color: red")
                return False

        except Exception as e:
            self.status_label.setText(f"Статус: Ошибка подключения ({str(e)})")
            self.status_label.setStyleSheet("color: red")
            return False

    def get_settings(self):
        return self.ip_input.text(), self.port_input.text()


class CameraSettingsDialog(QDialog):
    def __init__(self, camera_urls, usb_enabled, usb_camera_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки камер")
        self.layout = QVBoxLayout(self)

        # Camera 1 settings
        self.camera1_group = QGroupBox("Камера 1", self)
        self.camera1_layout = QFormLayout()
        self.camera1_url = QLineEdit(
            camera_urls[0] if camera_urls[0] else "http://192.168.1.106:4747/mjpegfeed?960x720", self)
        self.camera1_layout.addRow("URL:", self.camera1_url)
        self.camera1_enabled = QCheckBox("Активна", self)
        self.camera1_enabled.setChecked(bool(camera_urls[0]))
        self.camera1_layout.addRow(self.camera1_enabled)
        self.camera1_group.setLayout(self.camera1_layout)
        self.layout.addWidget(self.camera1_group)

        # Camera 2 settings
        self.camera2_group = QGroupBox("Камера 2", self)
        self.camera2_layout = QFormLayout()
        self.camera2_url = QLineEdit(camera_urls[1] if camera_urls[1] else "http://192.168.1.120:4747/video?960x720",
                                     self)
        self.camera2_layout.addRow("URL:", self.camera2_url)
        self.camera2_enabled = QCheckBox("Активна", self)
        self.camera2_enabled.setChecked(bool(camera_urls[1]))
        self.camera2_layout.addRow(self.camera2_enabled)
        self.camera2_group.setLayout(self.camera2_layout)
        self.layout.addWidget(self.camera2_group)

        # USB Camera
        self.usb_camera_group = QGroupBox("USB Камера", self)
        self.usb_camera_layout = QFormLayout()
        self.usb_camera_enabled = QCheckBox("Активна", self)
        self.usb_camera_enabled.setChecked(usb_enabled)
        self.usb_camera_layout.addRow(self.usb_camera_enabled)

        # Buttons
        self.buttons = QHBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Отмена", self)
        self.cancel_button.clicked.connect(self.reject)
        self.buttons.addWidget(self.ok_button)
        self.buttons.addWidget(self.cancel_button)
        self.layout.addLayout(self.buttons)

        # Добавляем выбор индекса USB-камеры
        self.usb_camera_index_combo = QComboBox(self)
        available_cameras = get_available_usb_cameras()
        for i in available_cameras:
            self.usb_camera_index_combo.addItem(f"Камера {i}", i)
        if usb_camera_index in available_cameras:
            index = available_cameras.index(usb_camera_index)
            self.usb_camera_index_combo.setCurrentIndex(index)
        self.usb_camera_layout.addRow("Выберите USB камеру:", self.usb_camera_index_combo)

        self.usb_camera_group.setLayout(self.usb_camera_layout)
        self.layout.addWidget(self.usb_camera_group)

    def get_settings(self):
        camera1_url = self.camera1_url.text() if self.camera1_enabled.isChecked() else ""
        camera2_url = self.camera2_url.text() if self.camera2_enabled.isChecked() else ""
        usb_enabled = self.usb_camera_enabled.isChecked()
        usb_camera_index = self.usb_camera_index_combo.currentData()
        return [camera1_url, camera2_url], usb_enabled, usb_camera_index


class VideoApp(QWidget):
    connection_status_changed = pyqtSignal(str)
    camera_connected_changed = pyqtSignal(bool)

    def __init__(self, coco_model, license_plate_detector, mot_tracker, vehicles, get_car, read_license_plate,
                 insert_car_data):
        super().__init__()
        self.coco_model = coco_model
        self.license_plate_detector = license_plate_detector
        self.mot_tracker = mot_tracker
        self.vehicles = vehicles
        self.get_car = get_car
        self.read_license_plate = read_license_plate
        self.insert_car_data = insert_car_data
        self.usb_camera_index = 0

        # Добавляем переменную для хранения текущего изображения
        self.best_text = None
        self.best_score = 0.0
        self.last_direction = None

        self.last_recognized_plate = None
        self.last_recognized_score = 0.0


        # Server settings
        self.server_ip = "192.168.1.159"
        self.server_port = "8765"
        self.connection_status = "disconnected"

        # Camera settings
        self.camera_urls = ["http://192.168.1.106:4747/mjpegfeed?960x720", "http://192.168.1.120:4747/video?960x720"]
        self.usb_enabled = False
        self.video_file = ""

        # Initialize UI
        self.init_ui()

        # Initialize variables
        self.is_streaming = False
        self.frame_queue = Queue(maxsize=2)
        self.streaming_thread = threading.Thread(target=self.streaming_worker, daemon=True)
        self.streaming_thread.start()

        self.cap1 = None
        self.cap2 = None
        self.video_cap = None
        self.usb_cap = None
        self.fps_limit = 30
        self.show_fps = True
        self.current_camera = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.paused = False
        self.recognized_plates = set()
        self.frame_times = []

        # Connect signals
        self.connection_status_changed.connect(self.update_connection_status)
        self.camera_connected_changed.connect(self.update_camera_button_status)

    def init_ui(self):
        self.setWindowTitle("Vehicle & License Plate Recognition")
        self.setGeometry(100, 100, 800, 600)

        # Create main widgets
        self.video_label = QLabel("Нажмите 'Подключить' для запуска камеры", self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.plate_text_display = QTextEdit(self)
        self.plate_text_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.plate_text_display.setReadOnly(True)

        # Create bottom panel with minimal buttons
        self.bottom_panel = QHBoxLayout()

        self.pause_button = QPushButton("Пауза", self)
        self.pause_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        self.bottom_panel.addWidget(self.pause_button)

        self.camera_switch_button = QPushButton("Переключить камеру", self)
        self.camera_switch_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.camera_switch_button.clicked.connect(self.switch_camera)
        self.camera_switch_button.setEnabled(False)
        self.bottom_panel.addWidget(self.camera_switch_button)

        self.connect_button = QPushButton("Подключить", self)
        self.connect_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.connect_button.clicked.connect(self.connect_cameras)
        self.bottom_panel.addWidget(self.connect_button)

        # Create main layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label, 8)
        layout.addWidget(self.plate_text_display, 2)
        layout.addLayout(self.bottom_panel, 1)
        self.setLayout(layout)

        # Create menu bar with dropdown menus
        self.menu_bar = QMenuBar(self)

        # File menu
        self.file_menu = QMenu("Файл", self.menu_bar)
        self.load_video_action = QAction("Загрузить видео", self)
        self.load_video_action.triggered.connect(self.load_video_file)
        self.file_menu.addAction(self.load_video_action)

        # Добавляем пункт для загрузки изображений
        self.load_image_action = QAction("Загрузить изображение(я)", self)
        self.load_image_action.triggered.connect(self.load_image_files)
        self.file_menu.addAction(self.load_image_action)

        # Добавляем пункты для навигации по изображениям
        self.next_image_action = QAction("Следующее изображение", self)
        self.next_image_action.triggered.connect(self.show_next_image)
        self.file_menu.addAction(self.next_image_action)

        self.prev_image_action = QAction("Предыдущее изображение", self)
        self.prev_image_action.triggered.connect(self.show_prev_image)
        self.file_menu.addAction(self.prev_image_action)

        self.exit_action = QAction("Выход", self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)
        self.menu_bar.addMenu(self.file_menu)

        # Camera menu
        self.camera_menu = QMenu("Камеры", self.menu_bar)
        self.camera_settings_action = QAction("Настроить камеры", self)
        self.camera_settings_action.triggered.connect(self.show_camera_settings)
        self.camera_menu.addAction(self.camera_settings_action)

        self.select_camera_action = QAction("Выбрать камеру", self)
        self.select_camera_action.triggered.connect(self.select_camera)
        self.camera_menu.addAction(self.select_camera_action)
        self.menu_bar.addMenu(self.camera_menu)

        # Stream menu
        self.stream_menu = QMenu("Трансляция", self.menu_bar)
        self.stream_settings_action = QAction("Настройки сервера", self)
        self.stream_settings_action.triggered.connect(self.show_server_settings)
        self.stream_menu.addAction(self.stream_settings_action)

        self.stream_action = QAction("Трансляция на сервер", self, checkable=True)
        self.stream_action.triggered.connect(self.toggle_streaming)
        self.stream_menu.addAction(self.stream_action)
        self.menu_bar.addMenu(self.stream_menu)

        # Display menu
        self.display_menu = QMenu("Отображение", self.menu_bar)
        self.fps_action = QAction("Ограничить FPS", self)
        self.fps_action.triggered.connect(self.set_fps_limit)
        self.display_menu.addAction(self.fps_action)

        self.show_fps_action = QAction("Показать FPS", self, checkable=True)
        self.show_fps_action.setChecked(True)
        self.show_fps_action.triggered.connect(self.toggle_show_fps)
        self.display_menu.addAction(self.show_fps_action)
        self.menu_bar.addMenu(self.display_menu)

        # Help menu
        self.help_menu = QMenu("Помощь", self.menu_bar)
        self.about_action = QAction("О программе", self)
        self.about_action.triggered.connect(self.show_about)
        self.help_menu.addAction(self.about_action)
        self.menu_bar.addMenu(self.help_menu)

        self.layout().setMenuBar(self.menu_bar)

        # Create status bar
        self.status_bar = QHBoxLayout()
        self.connection_status_indicator = StatusIndicator(self)
        self.connection_status_label = QLabel("Трансляция offline", self)

        self.status_bar.addWidget(self.connection_status_indicator)
        self.status_bar.addWidget(self.connection_status_label)
        self.status_bar.addStretch()

        layout.addLayout(self.status_bar)

    def load_image_files(self):
        """Загрузка одного или нескольких изображений"""
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображение(я)", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options
        )

        if files:
            self.image_files = files
            self.current_image_index = 0
            self.load_and_process_image(self.image_files[0])
            self.current_camera = "Изображение"
            self.update_camera_buttons()

    def load_and_process_image(self, file_path):
        """Загрузка и обработка одного изображения"""
        try:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                # Останавливаем таймер, если он активен
                if self.timer.isActive():
                    self.timer.stop()

                # Обрабатываем изображение
                results = model_prediction(
                    self.current_image,
                    self.coco_model,
                    self.license_plate_detector,
                    reader
                )

                if len(results) == 3:
                    # Найдены и распознаны номера
                    prediction, texts, license_plate_crop = results
                    self.display_processed_image(prediction)
                    self.plate_text_display.clear()
                    for text in texts:
                        if text:
                            self.plate_text_display.append(text)
                elif len(results) == 2:
                    # Либо номера не найдены, либо текст не распознан
                    prediction, messages = results
                    self.display_processed_image(prediction)
                    self.plate_text_display.clear()
                    for msg in messages:
                        self.plate_text_display.append(msg)
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при обработке изображения: {str(e)}")

    def display_processed_image(self, image):
        """Отображение обработанного изображения"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def show_next_image(self):
        """Показать следующее изображение из списка"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_and_process_image(self.image_files[self.current_image_index])

    def show_prev_image(self):
        """Показать предыдущее изображение из списка"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_and_process_image(self.image_files[self.current_image_index])

    def update_connection_status(self, status):
        self.connection_status = status
        self.connection_status_indicator.set_status(status)

        if status == "connected":
            self.connection_status_label.setText("Трансляция online")
            if self.is_streaming:
                self.stream_action.setChecked(True)
        elif status == "connecting":
            self.connection_status_label.setText("Подключение...")
        else:
            self.connection_status_label.setText("Трансляция offline")
            self.stream_action.setChecked(False)

    def update_camera_button_status(self, is_connected):
        if is_connected:
            self.connect_button.setStyleSheet("background-color: green")
        else:
            self.connect_button.setStyleSheet("")

    def show_server_settings(self):
        try:
            logging.info("Opening server settings dialog")
            dialog = ServerSettingsDialog(self)
            if dialog.exec_():
                self.server_ip, self.server_port = dialog.get_settings()
                logging.info(f"Server settings updated: {self.server_ip}:{self.server_port}")
                # self.test_server_connection()
        except Exception as e:
            logging.error(f"Error in server settings dialog: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка в настройках сервера: {str(e)}")

    def test_server_connection(self):
        def connection_test():
            try:
                self.connection_status_changed.emit("connecting")
                logging.info("Testing server connection...")

                # Test basic connection
                # test_response = requests.get(f"http://{self.server_ip}:{self.server_port}/test", timeout=2)
                # if test_response.status_code != 200:
                #     self.connection_status_changed.emit("disconnected")
                #     logging.error(f"Server test endpoint error: {test_response.status_code}")
                #     return False

                # Test upload endpoint
                upload_response = requests.post(
                    f"http://{self.server_ip}:{self.server_port}/upload",
                    json={"frame": "test"},
                    timeout=2
                )

                if upload_response.status_code in [200, 405]:  # 405 means endpoint exists
                    self.connection_status_changed.emit("connected")
                    logging.info("Server connection successful")
                    return True
                else:
                    self.connection_status_changed.emit("disconnected")
                    logging.error(f"Upload endpoint error: {upload_response.status_code}")
                    return False

            except Exception as e:
                self.connection_status_changed.emit("disconnected")
                logging.error(f"Connection test failed: {str(e)}")
                return False

        threading.Thread(target=connection_test, daemon=True).start()

    def show_camera_settings(self):
        try:
            logging.info("Opening camera settings dialog")
            dialog = CameraSettingsDialog(self.camera_urls, self.usb_enabled, self.usb_camera_index, self)
            if dialog.exec_():
                self.camera_urls, self.usb_enabled, self.usb_camera_index = dialog.get_settings()
                logging.info(
                    f"Camera settings updated: Camera1={self.camera_urls[0]}, Camera2={self.camera_urls[1]}, "
                    f"USB={self.usb_enabled}, USB Index={self.usb_camera_index}")

                # Reset current camera if settings changed
                if self.current_camera:
                    if (self.current_camera == "Камера 1" and not self.camera_urls[0]) or \
                            (self.current_camera == "Камера 2" and not self.camera_urls[1]) or \
                            (self.current_camera == "USB Камера" and not self.usb_enabled):
                        self.current_camera = None
                        self.timer.stop()
                        self.release_cameras()
                        self.video_label.setText("Настройки камер изменены. Выберите камеру снова.")
                        self.connect_button.setText("Подключить")
                        self.pause_button.setEnabled(False)
                        self.camera_switch_button.setEnabled(False)
                        self.camera_connected_changed.emit(False)
        except Exception as e:
            logging.error(f"Error in camera settings dialog: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка в настройках камер: {str(e)}")

    def select_camera(self):
        try:
            logging.info("Selecting camera")
            items = []
            if self.camera_urls[0]:
                items.append("Камера 1")
            if self.camera_urls[1]:
                items.append("Камера 2")
            if self.usb_enabled:
                items.append(f"USB Камера (индекс {self.usb_camera_index})")
            if self.video_file:
                items.append("Видеофайл")

            if not items:
                QMessageBox.warning(self, "Ошибка", "Нет доступных камер. Настройте камеры в меню 'Настроить камеры'")
                return

            item, ok = QInputDialog.getItem(self, "Выбор камеры", "Выберите камеру:", items, 0, False)
            if ok and item:
                # Убираем уточнение про индекс для USB-камеры
                if item.startswith("USB Камера"):
                    self.current_camera = "USB Камера"
                else:
                    self.current_camera = item
                logging.info(f"Selected camera: {self.current_camera}")
        except Exception as e:
            logging.error(f"Error in select camera: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при выборе камеры: {str(e)}")

    def update_camera_buttons(self):
        if self.current_camera in ["Камера 1", "Камера 2"]:
            self.camera_switch_button.setEnabled(True)
            self.camera_switch_button.setText("Переключить камеру")
        else:
            self.camera_switch_button.setEnabled(False)

        if self.current_camera:
            self.connect_button.setText("Отключить")
        else:
            self.connect_button.setText("Подключить")

    def switch_camera(self):
        try:
            logging.info("Switching camera")
            if self.current_camera == "Камера 1" and self.camera_urls[1]:
                self.current_camera = "Камера 2"
            elif self.current_camera == "Камера 2" and self.camera_urls[0]:
                self.current_camera = "Камера 1"

            logging.info(f"Switched to camera: {self.current_camera}")
            self.restart_video_streams()
        except Exception as e:
            logging.error(f"Error switching camera: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при переключении камеры: {str(e)}")

    def connect_cameras(self):
        try:
            logging.info("Connecting/disconnecting cameras")
            if not self.current_camera:
                # self.select_camera()
                return

            if self.timer.isActive():
                # Disconnect cameras
                self.timer.stop()
                self.release_cameras()
                self.video_label.setText("Камеры отключены")
                self.connect_button.setText("Подключить")
                self.pause_button.setEnabled(False)
                self.camera_switch_button.setEnabled(False)
                self.camera_connected_changed.emit(False)
                logging.info("Cameras disconnected")
            else:
                # Connect cameras
                if self.restart_video_streams():
                    self.timer.start(1000 // self.fps_limit)
                    self.connect_button.setText("Отключить")
                    self.pause_button.setEnabled(True)
                    self.camera_connected_changed.emit(True)
                    logging.info("Cameras connected")
                else:
                    self.video_label.setText("Ошибка подключения к камере")
                    QMessageBox.warning(self, "Ошибка", "Не удалось подключиться к камере")
        except Exception as e:
            logging.error(f"Error connecting cameras: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при подключении камер: {str(e)}")

    def release_cameras(self):
        try:
            if self.cap1 and self.cap1.isOpened():
                self.cap1.release()
            if self.cap2 and self.cap2.isOpened():
                self.cap2.release()
            if self.video_cap and self.video_cap.isOpened():
                self.video_cap.release()
            if hasattr(self, 'usb_cap') and self.usb_cap and self.usb_cap.isOpened():
                self.usb_cap.release()
        except Exception as e:
            logging.error(f"Error releasing cameras: {e}")

        self.cap1 = None
        self.cap2 = None
        self.video_cap = None
        self.usb_cap = None

    def restart_video_streams(self):
        self.release_cameras()

        try:
            if self.current_camera == "Камера 1" and self.camera_urls[0]:
                self.cap1 = cv2.VideoCapture(self.camera_urls[0])
                if not self.cap1.isOpened():
                    logging.error(f"Error opening video stream for camera 1: {self.camera_urls[0]}")
                    self.cap1 = None
                    return False
                logging.info(f"Successfully connected to camera 1: {self.camera_urls[0]}")
                return True

            elif self.current_camera == "Камера 2" and self.camera_urls[1]:
                self.cap2 = cv2.VideoCapture(self.camera_urls[1])
                if not self.cap2.isOpened():
                    logging.error(f"Error opening video stream for camera 2: {self.camera_urls[1]}")
                    self.cap2 = None
                    return False
                logging.info(f"Successfully connected to camera 2: {self.camera_urls[1]}")
                return True

            elif self.current_camera == "USB Камера" and self.usb_enabled:
                self.usb_cap = cv2.VideoCapture(self.usb_camera_index)
                if not self.usb_cap.isOpened():
                    logging.error(f"Error opening USB camera with index {self.usb_camera_index}")
                    self.usb_cap = None
                    return False
                logging.info(f"Successfully connected to USB camera with index {self.usb_camera_index}")
                return True

            elif self.current_camera == "Видеофайл" and self.video_file:
                self.video_cap = cv2.VideoCapture(self.video_file)
                if not self.video_cap.isOpened():
                    logging.error(f"Error opening video file: {self.video_file}")
                    self.video_cap = None
                    return False
                logging.info(f"Successfully opened video file: {self.video_file}")
                return True
        except Exception as e:
            logging.error(f"Error connecting to camera: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка подключения к камере: {str(e)}")
            return False

        return False

    def load_video_file(self):
        try:
            logging.info("Loading video file")
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Загрузить видео", "",
                "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
                options=options
            )
            if file_name:
                self.video_file = file_name
                self.current_camera = "Видеофайл"
                logging.info(f"Video file selected: {file_name}")
                self.update_camera_buttons()
                if not self.timer.isActive():
                    self.connect_cameras()
        except Exception as e:
            logging.error(f"Error loading video file: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке видео: {str(e)}")

    def show_about(self):
        QMessageBox.about(self, "О программе",
                          "Vehicle & License Plate Recognition System\n\n"
                          "Версия 1.0\n"
                          "Разработано для автоматического распознавания номеров автомобилей")

    def toggle_streaming(self):
        try:
            logging.info("Toggling streaming")
            if self.stream_action.isChecked():
                # При включении трансляции сначала проверяем подключение
                # if self.connection_status != "connected":
                #     self.test_server_connection()
                #     if self.connection_status != "connected":
                #         self.stream_action.setChecked(False)
                #         return
                self.start_streaming()
            else:
                self.stop_streaming()
        except Exception as e:
            logging.error(f"Error toggling streaming: {e}")
            self.stream_action.setChecked(False)
            QMessageBox.critical(self, "Ошибка", f"Ошибка при переключении трансляции: {str(e)}")

    def start_streaming(self):
        if not self.is_streaming:
            self.is_streaming = True
            logging.info("Трансляция на сервер запущена")
            self.connection_status_changed.emit("connected")

    def stop_streaming(self):
        if self.is_streaming:
            self.is_streaming = False
            logging.info("Трансляция на сервер остановлена")
            if self.connection_status == "connected":
                self.connection_status_changed.emit("disconnected")

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

    def set_fps_limit(self):
        try:
            logging.info("Setting FPS limit")
            fps_limit, ok = QInputDialog.getInt(
                self, "Ограничить FPS", "Введите максимальное FPS:",
                self.fps_limit, 1, 60, 1
            )
            if ok:
                self.fps_limit = fps_limit
                if self.timer.isActive():
                    self.timer.setInterval(1000 // self.fps_limit)
                logging.info(f"FPS limit set to {self.fps_limit}")
        except Exception as e:
            logging.error(f"Error setting FPS limit: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при установке FPS: {str(e)}")

    def toggle_show_fps(self):
        self.show_fps = self.show_fps_action.isChecked()
        logging.info(f"Show FPS: {self.show_fps}")

    def toggle_pause(self):
        try:
            logging.info("Toggling pause")
            if self.paused:
                self.timer.start(1000 // self.fps_limit)
                self.pause_button.setText("Пауза")
                logging.info("Video resumed.")
            else:
                self.timer.stop()
                self.pause_button.setText("Продолжить")
                logging.info("Video paused.")
            self.paused = not self.paused
        except Exception as e:
            logging.error(f"Error toggling pause: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при переключении паузы: {str(e)}")

    def update_frame(self):
        try:
            if not any([self.cap1, self.cap2, self.video_cap, hasattr(self, 'usb_cap')]):
                return

            start_time = time.time()
            ret, frame = self.get_current_frame()
            if not ret or frame is None:
                return

            # frame = cv2.resize(frame, (1920, 1080))
            results = model_prediction(frame, self.coco_model, self.license_plate_detector, reader)

            # Обработка всех обнаруженных номеров
            if len(results) >= 6:  # Теперь возвращается 6 значений
                processed_frame, texts, crops, current_best_text, current_best_score, direction = results

                # Обновляем лучший результат
                if current_best_score > self.best_score:
                    self.best_text = current_best_text
                    self.best_score = current_best_score
                    self.last_direction = direction
                    logging.info(
                        f"Новый лучший результат: {self.best_text} ({self.best_score:.2f}), направление: {direction}")

                # Новая логика вывода номеров
                if current_best_score > 0.85:
                    if self.last_recognized_plate != current_best_text:
                        display_text = f"{current_best_text} ({current_best_score:.2f})"
                        if direction:
                            display_text += f" {direction}"
                        self.plate_text_display.append(display_text)
                        self.last_recognized_plate = current_best_text
                        self.last_recognized_score = current_best_score

                        _, buffer = cv2.imencode('.jpg', processed_frame)
                        if buffer is not None:
                            photo_bytes = buffer.tobytes()
                            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            car_type = "Car"  # Можно определить более точно из результатов детекции
                            self.insert_car_data(
                                self.last_recognized_plate,
                                photo_bytes,
                                car_type,
                                current_date
                            )

                        logging.info(f"Отображен новый номер: {current_best_text}, направление: {direction}")

            # Отображение FPS
            if self.show_fps:
                current_fps = self.calculate_fps(start_time)
                processed_frame = draw_license_plate_text(
                    processed_frame,
                    f"FPS: {current_fps:.2f}",
                    (10, 10)
                )

            # Всегда отображаем лучший результат, если он есть
            if self.last_recognized_plate and self.last_recognized_score > 0.85:
                display_text = f"{self.last_recognized_plate} ({self.last_recognized_score:.2f})"
                if self.last_direction:
                    display_text += f" {self.last_direction}"

                processed_frame = draw_best_result(
                    processed_frame,
                    display_text,
                    self.last_recognized_score,
                    (350, 10)
                )

            self.display_processed_frame(processed_frame)

            if self.is_streaming:
                self.send_frame_to_stream(processed_frame)

        except Exception as e:
            logging.error(f"Error in update_frame: {e}")

    # def update_frame(self):
    #     try:
    #         if not any([self.cap1, self.cap2, self.video_cap, hasattr(self, 'usb_cap')]):
    #             return
    #
    #         start_time = time.time()
    #         ret, frame = self.get_current_frame()
    #         if not ret or frame is None:
    #             return
    #
    #         frame = cv2.resize(frame, (1920, 1080))
    #         results = model_prediction(frame, self.coco_model, self.license_plate_detector, reader)
    #
    #         # Обработка всех обнаруженных номеров
    #         if len(results) >= 6:  # Теперь возвращается 6 значений
    #             processed_frame, texts, crops, current_best_text, current_best_score, direction = results
    #
    #             # Обновляем лучший результат
    #             if current_best_score > getattr(self, 'best_score', 0):
    #                 self.best_text = current_best_text
    #                 self.best_score = current_best_score
    #                 self.last_direction = direction  # Сохраняем направление
    #                 logging.info(
    #                     f"Новый лучший результат: {self.best_text} ({self.best_score:.2f}), направление: {direction}")
    #
    #             # Новая логика вывода номеров
    #             if current_best_score > 0.85:
    #                 if self.last_recognized_plate != current_best_text:
    #                     # Это новый номер, отличающийся от предыдущего
    #                     display_text = f"{current_best_text} ({current_best_score:.2f})"
    #                     if direction:
    #                         display_text += f" ({direction})"
    #                     self.plate_text_display.append(display_text)
    #                     self.last_recognized_plate = current_best_text
    #                     self.last_recognized_score = current_best_score
    #                     logging.info(f"Отображен новый номер: {current_best_text}, направление: {direction}")
    #
    #         # Отображение FPS
    #         if self.show_fps:
    #             current_fps = self.calculate_fps(start_time)
    #             processed_frame = draw_license_plate_text(
    #                 processed_frame,
    #                 f"FPS: {current_fps:.2f}",
    #                 (10, 10)
    #             )
    #
    #         # Отображение лучшего результата с направлением
    #         if hasattr(self, 'best_text') and hasattr(self, 'best_score'):
    #             if self.best_score > 0.85:
    #                 display_text = f"{self.best_text} ({self.best_score:.2f})"
    #                 if hasattr(self, 'last_direction') and self.last_direction:
    #                     display_text += f" {self.last_direction}"
    #
    #                 processed_frame = draw_best_result(
    #                     processed_frame,
    #                     display_text,
    #                     self.best_score,
    #                     (10, 350)
    #                 )
    #
    #         self.display_processed_frame(processed_frame)
    #
    #         if self.is_streaming:
    #             self.send_frame_to_stream(processed_frame)
    #
    #     except Exception as e:
    #         logging.error(f"Error in update_frame: {e}")

    def get_current_frame(self):
        """Получает текущий кадр из активного источника"""
        if self.current_camera == "Камера 1" and self.cap1 and self.cap1.isOpened():
            return self.cap1.read()
        elif self.current_camera == "Камера 2" and self.cap2 and self.cap2.isOpened():
            return self.cap2.read()
        elif self.current_camera == "USB Камера" and hasattr(self, 'usb_cap') and self.usb_cap.isOpened():
            return self.usb_cap.read()
        elif self.current_camera == "Видеофайл" and self.video_cap and self.video_cap.isOpened():
            return self.video_cap.read()
        return False, None

    def process_recognized_texts(self, texts):
        """Обрабатывает распознанные тексты номеров"""
        for text in texts:
            if text and text not in self.recognized_plates:
                self.recognized_plates.add(text)
                self.plate_text_display.append(text)

    def calculate_fps(self, start_time):
        """Вычисляет текущий FPS"""
        end_time = time.time()
        self.frame_times.append(end_time - start_time)
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)
        return len(self.frame_times) / sum(self.frame_times)

    def display_processed_frame(self, frame):
        """Отображает обработанный кадр в интерфейсе"""
        height, width, channel = frame.shape
        q_img = QImage(frame.data, width, height, width * channel, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def send_frame_to_stream(self, frame):
        """Отправляет кадр на сервер трансляции"""
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def streaming_worker(self):
        while True:
            try:
                frame = self.frame_queue.get()
                if frame is None:
                    break
                self.send_frame(frame)
            except Exception as e:
                logging.error(f"Error in streaming worker: {e}")

    def closeEvent(self, event):
        try:
            logging.info("Closing application")
            self.stop_streaming()
            self.frame_queue.put(None)
            if self.streaming_thread.is_alive():
                self.streaming_thread.join()
            self.release_cameras()
            logging.info("Application closed.")
        except Exception as e:
            logging.error(f"Error during close: {e}")
        event.accept()