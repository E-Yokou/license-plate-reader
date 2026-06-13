# ui.py
import logging
import time
from collections import deque

import numpy as np
import torch
from PyQt5.QtWidgets import (
    QWidget, QLabel, QSizePolicy, QPushButton, QVBoxLayout, QApplication,
    QMenuBar, QMenu, QAction, QInputDialog, QLineEdit, QCheckBox, QDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QFileDialog, QMessageBox, QComboBox,
    QGridLayout, QListWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PIL import ImageFont, ImageDraw, Image
import cv2
from util import model_prediction, reader, draw_best_result, draw_tracked_plate, license_complies_format, \
    read_license_plate, get_plate_center, get_car_center, is_plate_inside_car, draw_tracking_info
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
    camera_connected_changed = pyqtSignal(bool)

    def __init__(self, coco_model, license_plate_detector, mot_tracker, vehicles, get_car, read_license_plate):
        super().__init__()
        self.coco_model = coco_model
        self.license_plate_detector = license_plate_detector
        self.mot_tracker = mot_tracker
        self.vehicles = vehicles
        self.get_car = get_car
        self.read_license_plate = read_license_plate
        self.usb_camera_index = 0

        self.best_text = None
        self.best_score = 0.0
        self.last_direction = None
        self.last_recognized_plate = None
        self.last_recognized_score = 0.0

        # Camera settings
        self.camera_urls = ["", ""]
        self.usb_enabled = False
        self.video_file = ""
        self.current_camera_url = None

        # Initialize UI
        self.init_ui()

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

        self.tracked_plates = {}  # {track_id: {'plate_text': str, 'plate_score': float, 'last_seen': float}}
        self.plate_history = {}
        self.recent_plates = deque(maxlen=10)  # последние 10 уникальных номеров для панели

        self.camera_names = []

        self.camera_connected_changed.connect(self.update_camera_button_status)

    def init_ui(self):
        self.setWindowTitle("Vehicle & License Plate Recognition")
        self.setGeometry(100, 100, 1100, 650)

        # Video grid
        self.video_labels = []
        self.video_layout = QGridLayout()

        # Bottom panel — кнопки фиксированной высоты
        self.bottom_panel = QHBoxLayout()

        self.pause_button = QPushButton("Пауза", self)
        self.pause_button.setFixedHeight(36)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        self.bottom_panel.addWidget(self.pause_button)

        self.connect_button = QPushButton("Подключить", self)
        self.connect_button.setFixedHeight(36)
        self.connect_button.clicked.connect(self.connect_cameras)
        self.bottom_panel.addWidget(self.connect_button)

        self.camera_switch_button = QPushButton("Переключить камеру", self)
        self.camera_switch_button.setFixedHeight(36)
        self.camera_switch_button.clicked.connect(self.switch_camera)
        self.camera_switch_button.setEnabled(False)
        self.bottom_panel.addWidget(self.camera_switch_button)

        # Оборачиваем кнопки в виджет с фиксированной высотой — гарантирует привязку к низу
        self.bottom_widget = QWidget()
        self.bottom_widget.setLayout(self.bottom_panel)
        self.bottom_widget.setFixedHeight(50)

        # Левая колонка: видео + кнопки (кнопки всегда внизу)
        left_layout = QVBoxLayout()
        left_layout.addLayout(self.video_layout, 1)
        left_layout.addWidget(self.bottom_widget, 0)

        # Правая панель: список распознанных номеров (стиль системный)
        right_layout = QVBoxLayout()
        plates_header = QLabel("Распознанные номера")
        plates_header.setAlignment(Qt.AlignCenter)
        plates_header.setStyleSheet("font-weight: bold; padding: 4px;")
        plates_header.setFixedHeight(28)
        right_layout.addWidget(plates_header)

        self.plates_list = QListWidget()
        self.plates_list.setFixedWidth(200)
        self.plates_list.setStyleSheet(
            "QListWidget::item { padding: 5px; }"
        )
        right_layout.addWidget(self.plates_list, 1)

        # Главный горизонтальный контент
        content_layout = QHBoxLayout()
        content_layout.addLayout(left_layout, 1)
        content_layout.addLayout(right_layout, 0)

        layout = QVBoxLayout()
        layout.addLayout(content_layout, 1)  # stretch=1 чтобы контент занимал всё окно
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

        # Settings menu
        self.settings_menu = QMenu("Настройки", self.menu_bar)

        # Threshold selection
        self.threshold_action = QAction("Порог распознавания", self)
        self.threshold_action.triggered.connect(self.set_recognition_threshold)
        self.settings_menu.addAction(self.threshold_action)

        # Добавляем пункт для отключения шаблонной обработки
        self.template_processing_action = QAction("Шаблоны номеров", self, checkable=True)
        self.template_processing_action.setChecked(True)  # Включено по умолчанию
        self.template_processing_action.triggered.connect(self.toggle_template_processing)
        self.settings_menu.addAction(self.template_processing_action)

        self.menu_bar.addMenu(self.settings_menu)

        # Добавил переменную для хранения текущего порога
        self.recognition_threshold = 0.85  # Значение по умолчанию

        # Camera menu
        self.camera_menu = QMenu("Камеры", self.menu_bar)

        self.configure_cameras_action = QAction("Настроить камеры...", self)
        self.configure_cameras_action.triggered.connect(self.show_camera_settings)
        self.camera_menu.addAction(self.configure_cameras_action)

        self.select_camera_action = QAction("Выбрать камеру", self)
        self.select_camera_action.triggered.connect(self.select_camera)
        self.camera_menu.addAction(self.select_camera_action)
        self.menu_bar.addMenu(self.camera_menu)

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

    def toggle_template_processing(self):
        """Включает/выключает обработку текста под шаблоны номерных знаков"""
        self.template_processing_enabled = self.template_processing_action.isChecked()
        logging.info(
            f"Обработка под шаблоны номеров: {'включена' if self.template_processing_enabled else 'выключена'}")

    def set_recognition_threshold(self):
        """Устанавливает порог вероятности для распознавания номеров"""
        try:
            threshold, ok = QInputDialog.getDouble(
                self,
                "Порог распознавания",
                "Введите порог вероятности (0.1-0.99):",
                self.recognition_threshold,
                0.1,
                0.99,
                2
            )
            if ok:
                self.recognition_threshold = threshold
                logging.info(f"Установлен новый порог распознавания: {threshold}")
        except Exception as e:
            logging.error(f"Ошибка при установке порога: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при установке порога: {str(e)}")

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
            if self.current_image is None:
                return

            if self.timer.isActive():
                self.timer.stop()

            results = model_prediction(
                self.current_image,
                self.coco_model,
                self.license_plate_detector,
                reader
            )

            prediction = results[0]  # RGB numpy array
            texts = results[1] if len(results) > 1 else []

            self.display_processed_image(prediction)

            # Показываем распознанные номера в боковой панели
            for item in texts:
                if item and isinstance(item, (list, tuple)):
                    plate_text = item[0] if item else None
                elif isinstance(item, str):
                    plate_text = item
                else:
                    plate_text = None
                if plate_text:
                    self.add_to_plates_list(plate_text)
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка при обработке изображения: {str(e)}")

    def display_processed_image(self, image):
        """Отображение обработанного изображения (RGB numpy array)"""
        # Создаём label если ещё нет
        if not self.video_labels:
            video_label = QLabel(self)
            video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            video_label.setAlignment(Qt.AlignCenter)
            self.video_labels.append((None, video_label))
            self.video_layout.addWidget(video_label, 0, 0)

        _, video_label = self.video_labels[0]
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        video_label.setPixmap(pixmap)

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

    def update_camera_button_status(self, is_connected):
        if is_connected:
            self.connect_button.setStyleSheet("background-color: green")
        else:
            self.connect_button.setStyleSheet("")

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
                        self.clear_video_streams()
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
            camera_types = []

            if self.camera_urls[0]:
                items.append("Камера 1 (Статическая)")
                camera_types.append(("static_camera", 0))
            if self.camera_urls[1]:
                items.append("Камера 2 (Статическая)")
                camera_types.append(("static_camera", 1))
            if self.usb_enabled:
                items.append(f"USB Камера (индекс {self.usb_camera_index})")
                camera_types.append(("usb_camera", self.usb_camera_index))
            if hasattr(self, 'video_file') and self.video_file:
                items.append("Видеофайл")
                camera_types.append(("video_file", 0))

            # Всегда показываем опцию ручного ввода URL
            items.append("Ввести URL камеры...")
            camera_types.append(("manual_url", None))

            item, ok = QInputDialog.getItem(self, "Выбор камеры", "Выберите камеру:", items, 0, False)
            if ok and item:
                selected_index = items.index(item)
                camera_type, camera_param = camera_types[selected_index]
                if camera_type == "static_camera":
                    self.current_camera = f"Камера {camera_param + 1}"
                    self.current_camera_url = self.camera_urls[camera_param]
                    logging.info(f"Selected camera: {self.current_camera}")
                    self.update_camera_buttons()
                elif camera_type == "usb_camera":
                    self.current_camera = "USB Камера"
                    self.usb_camera_index = camera_param
                    logging.info(f"Selected camera: {self.current_camera}")
                    self.update_camera_buttons()
                elif camera_type == "video_file":
                    self.current_camera = "Видеофайл"
                    logging.info(f"Selected camera: {self.current_camera}")
                    self.update_camera_buttons()
                elif camera_type == "manual_url":
                    self.show_camera_settings()
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
            if self.timer.isActive():
                self.timer.stop()
                self.release_cameras()
                self.connect_button.setText("Подключить")
                self.pause_button.setEnabled(False)
                self.camera_connected_changed.emit(False)
                self.clear_video_streams()
                logging.info("Cameras disconnected")
            else:
                if self.current_camera == "Видеофайл":
                    self.load_video()
                else:
                    self.add_cameras_manually()
        except Exception as e:
            logging.error(f"Error connecting cameras: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при подключении камер: {str(e)}")

    def add_cameras_manually(self):
        """Добавляет камеры вручную, если не удалось подключиться к MySQL"""
        try:
            # Запрашиваем количество камер
            num_cameras, ok = QInputDialog.getInt(
                self, "Ручной ввод камер", "Введите количество камер:", 1, 1, 4, 1
            )
            if not ok:
                return

            # Очищаем предыдущие данные
            self.release_cameras()
            self.clear_video_streams()
            self.camera_names = []

            camera_urls = []
            for i in range(num_cameras):
                url, ok = QInputDialog.getText(
                    self, f"Камера {i + 1}", f"Введите URL для камеры {i + 1}:",
                    QLineEdit.Normal, "http://192.168.1.106:4747/mjpegfeed?960x720"
                )
                if ok and url:
                    camera_urls.append(url)
                else:
                    return

            # Устанавливаем ручные URL камер
            self.camera_urls = camera_urls
            self.usb_enabled = False

            # Создаем временные камеры для отображения
            for i, url in enumerate(camera_urls):
                cap = self._open_capture(url)
                if cap.isOpened():
                    video_label = QLabel(self)
                    video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    video_label.setAlignment(Qt.AlignCenter)
                    self.video_labels.append((cap, video_label))
                    self.video_layout.addWidget(video_label, i // 2, i % 2)
                    self.camera_names.append(f"Камера {i + 1}")

            if self.video_labels:
                self.timer.start(1000 // self.fps_limit)
                self.connect_button.setText("Отключить")
                self.pause_button.setEnabled(True)
                self.camera_connected_changed.emit(True)
                logging.info(f"Ручные камеры подключены: {self.camera_urls}")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось подключиться ни к одной из указанных камер")
        except Exception as e:
            logging.error(f"Error in manual camera addition: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при ручном добавлении камер: {str(e)}")

    def clear_video_streams(self):
        """Clears all video displays by setting empty pixmaps"""
        for _, video_label in self.video_labels:
            video_label.clear()
            video_label.setText("Камера отключена")
            video_label.setStyleSheet("background-color: #333; color: white;")
            video_label.setAlignment(Qt.AlignCenter)

        for i in reversed(range(self.video_layout.count())):
            widget = self.video_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.video_labels.clear()

        self.best_text = None
        self.best_score = 0.0
        self.last_direction = None
        self.last_recognized_plate = None
        self.last_recognized_score = 0.0

    def release_cameras(self):
        try:
            for cap, _ in self.video_labels:
                if cap and cap.isOpened():
                    cap.release()
            self.video_labels.clear()
            if hasattr(self, 'video_cap') and self.video_cap:
                self.video_cap.release()
                self.video_cap = None
        except Exception as e:
            logging.error(f"Error releasing cameras: {e}")

    def determine_car_type(self, frame):
        """Определяет тип автомобиля по изображению"""
        # Ваша логика определения типа автомобиля
        # Например, можно использовать coco_model для классификации
        return "Car"  # Временная заглушка

    @staticmethod
    def _open_capture(source, timeout_ms=5000):
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.open(source)
        return cap

    def restart_video_streams(self):
        self.release_cameras()

        try:
            if self.current_camera.startswith("Камера из БД") and hasattr(self, 'current_camera_url'):
                self.cap1 = self._open_capture(self.current_camera_url)
                if not self.cap1.isOpened():
                    logging.error(f"Error opening video stream from DB: {self.current_camera_url}")
                    self.cap1 = None
                    return False
                return True

            elif self.current_camera == "Камера 1" and self.camera_urls[0]:
                self.cap1 = self._open_capture(self.camera_urls[0])
                if not self.cap1.isOpened():
                    self.cap1 = None
                    return False
                return True

            elif self.current_camera == "Камера 2" and self.camera_urls[1]:
                self.cap2 = self._open_capture(self.camera_urls[1])
                if not self.cap2.isOpened():
                    self.cap2 = None
                    return False
                return True

            elif self.current_camera == "USB Камера" and self.usb_enabled:
                self.usb_cap = self._open_capture(self.usb_camera_index)
                if not self.usb_cap.isOpened():
                    self.usb_cap = None
                    return False
                return True

            elif self.current_camera == "Видеофайл" and self.video_file:
                self.video_cap = cv2.VideoCapture(self.video_file)
                if not self.video_cap.isOpened():
                    self.video_cap = None
                    return False
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
            if file_name and os.path.exists(file_name):
                self.video_file = file_name
                self.current_camera = "Видеофайл"
                logging.info(f"Video file selected: {file_name}")
                self.update_camera_buttons()
                if not self.timer.isActive():
                    self.connect_cameras()
            else:
                logging.error("Selected video file does not exist.")
                QMessageBox.critical(self, "Ошибка", "Выбранный видеофайл не существует.")
        except Exception as e:
            logging.error(f"Error loading video file: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке видео: {str(e)}")

    def load_video(self):
        try:
            logging.info("Loading video file for playback")
            if not os.path.exists(self.video_file):
                logging.error(f"Video file does not exist: {self.video_file}")
                QMessageBox.critical(self, "Ошибка", f"Видеофайл не существует: {self.video_file}")
                return

            self.video_cap = cv2.VideoCapture(self.video_file)
            if not self.video_cap.isOpened():
                logging.error(f"Error opening video file: {self.video_file}")
                self.video_cap = None
                QMessageBox.critical(self, "Ошибка", f"Ошибка открытия видеофайла: {self.video_file}")
                return

            self.video_labels.clear()
            video_label = QLabel(self)
            video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            video_label.setAlignment(Qt.AlignCenter)
            self.video_labels.append((self.video_cap, video_label))
            self.video_layout.addWidget(video_label, 0, 0)

            self.timer.start(1000 // self.fps_limit)
            self.connect_button.setText("Отключить")
            self.pause_button.setEnabled(True)
            self.camera_connected_changed.emit(True)
            logging.info("Video file loaded and playback started")
        except Exception as e:
            logging.error(f"Error loading video for playback: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке видео для воспроизведения: {str(e)}")

    def show_about(self):
        QMessageBox.about(self, "О программе",
                          "Vehicle & License Plate Recognition System\n\n"
                          "Версия 1.0\n"
                          "Разработано для автоматического распознавания номеров автомобилей")

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
            if not self.video_labels:
                return

            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 24)
                font_small = ImageFont.truetype("DejaVuSans.ttf", 18)
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
                logging.warning("DejaVuSans.ttf not found, using default font")

            for idx, (cap, video_label) in enumerate(self.video_labels):
                start_time = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                camera_name = self.camera_names[idx] if idx < len(self.camera_names) else f"Камера {idx + 1}"

                # Конвертируем в PIL изображение для работы со шрифтами
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                # Детекция транспортных средств
                vehicle_detections = self.coco_model(frame)[0]
                vehicle_boxes = []
                for detection in vehicle_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in [2, 3, 5, 7]:  # Только автомобили
                        vehicle_boxes.append([x1, y1, x2, y2, score])
                        # Рисуем bounding box автомобиля
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

                # Трекинг транспортных средств
                track_ids = self.mot_tracker.update(np.asarray(vehicle_boxes)) if vehicle_boxes else []
                current_time = time.time()

                # Детекция номерных знаков
                license_detections = self.license_plate_detector(frame)[0]

                # Рисуем bbox всех обнаруженных номеров (жёлтый)
                for lp in license_detections.boxes.data.tolist():
                    lx1, ly1, lx2, ly2, lscore, _ = lp
                    draw.rectangle([lx1, ly1, lx2, ly2], outline=(255, 220, 0), width=2)

                # 1. Обновляем существующие треки
                updated_vehicles = set()
                for track in track_ids:
                    xcar1, ycar1, xcar2, ycar2, track_id = track
                    car_bbox = (xcar1, ycar1, xcar2, ycar2)

                    if track_id in self.tracked_plates:
                        # Проверяем, есть ли новый номер для этого авто
                        new_plate = None
                        for lp in license_detections.boxes.data.tolist():
                            x1, y1, x2, y2, score, _ = lp
                            if is_plate_inside_car((x1, y1, x2, y2), car_bbox):
                                plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                plate_text, plate_score = read_license_plate(plate_crop)

                                if plate_text and plate_score >= self.recognition_threshold:
                                    new_plate = (plate_text, plate_score)
                                    break

                        # Обновляем или сохраняем существующий номер
                        if new_plate:
                            self.tracked_plates[track_id] = {
                                'plate_text': new_plate[0],
                                'plate_score': new_plate[1],
                                'last_seen': current_time
                            }
                        else:
                            self.tracked_plates[track_id]['last_seen'] = current_time

                        updated_vehicles.add(track_id)

                # 2. Обрабатываем новые номера для необновленных авто
                for lp in license_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, _ = lp
                    plate_bbox = (x1, y1, x2, y2)
                    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    plate_text, plate_score = read_license_plate(plate_crop)

                    if not plate_text or plate_score < self.recognition_threshold:
                        continue

                    # Ищем ближайший автомобиль без номера
                    best_match = None
                    min_distance = float('inf')

                    for track in track_ids:
                        xcar1, ycar1, xcar2, ycar2, track_id = track
                        if track_id in updated_vehicles:
                            continue

                        car_bbox = (xcar1, ycar1, xcar2, ycar2)
                        if is_plate_inside_car(plate_bbox, car_bbox):
                            distance = np.linalg.norm(
                                np.array(get_plate_center(plate_bbox)) -
                                np.array(get_car_center(car_bbox)))

                            if distance < min_distance:
                                min_distance = distance
                            best_match = track_id

                            if best_match:
                                self.tracked_plates[best_match] = {
                                    'plate_text': plate_text,
                                    'plate_score': plate_score,
                                    'last_seen': current_time
                                }
                            updated_vehicles.add(best_match)

                # 3. Визуализация номеров на автомобилях + обновление панели
                for track in track_ids:
                    x1, y1, x2, y2, track_id = track
                    plate_info = self.tracked_plates.get(track_id)

                    if plate_info:
                        plate_text = plate_info['plate_text']
                        text = f"{plate_text} ({plate_info['plate_score']:.2f})"
                        text_bbox = draw.textbbox((0, 0), text, font=font)

                        # Рисуем подложку
                        draw.rectangle(
                            [x1, y1 - (text_bbox[3] - text_bbox[1]) - 10,
                             x1 + (text_bbox[2] - text_bbox[0]) + 10, y1],
                            fill=(0, 0, 255))

                        # Рисуем текст номера
                        draw.text(
                            (x1 + 5, y1 - (text_bbox[3] - text_bbox[1]) - 5),
                            text,
                            font=font,
                            fill=(255, 255, 255))

                        # Добавляем в боковую панель
                        self.add_to_plates_list(plate_text)

                # 4. Очистка старых треков (>5 секунд без обновления)
                to_delete = [tid for tid, plate in self.tracked_plates.items()
                             if current_time - plate['last_seen'] > 5.0]
                for tid in to_delete:
                    del self.tracked_plates[tid]

                # Отображение FPS
                if self.show_fps:
                    fps = 1.0 / (time.time() - start_time)
                    fps_text = f"FPS: {fps:.2f}"
                    fps_bbox = draw.textbbox((0, 0), fps_text, font=font_small)
                    draw.rectangle(
                        [10, 10, 10 + (fps_bbox[2] - fps_bbox[0]) + 10, 10 + (fps_bbox[3] - fps_bbox[1]) + 10],
                        fill=(0, 0, 0, 128))
                    draw.text((15, 15), fps_text, font=font_small, fill=(0, 255, 0))

                # Отображение названия камеры
                name_bbox = draw.textbbox((0, 0), camera_name, font=font_small)
                draw.rectangle(
                    [10, 40, 10 + (name_bbox[2] - name_bbox[0]) + 10, 40 + (name_bbox[3] - name_bbox[1]) + 10],
                    fill=(0, 0, 0, 128))
                draw.text((15, 45), camera_name, font=font_small, fill=(255, 255, 255))

                self.display_processed_frame(pil_img, video_label)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error("CUDA memory error - trying to recover")
                torch.cuda.empty_cache()
                self.restart_video_streams()
            else:
                logging.error(f"Runtime error in update_frame: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in update_frame: {e}")
            self.release_cameras()
            self.timer.stop()

    def add_to_plates_list(self, plate_text):
        if plate_text in self.recent_plates:
            return
        self.recent_plates.append(plate_text)
        self.plates_list.insertItem(0, plate_text)  # новые — сверху
        while self.plates_list.count() > 10:
            self.plates_list.takeItem(self.plates_list.count() - 1)

    def clean_old_tracks(self):
        """Очищает треки, которые не обновлялись более 5 секунд"""
        current_time = time.time()
        to_delete = [tid for tid, plate in self.tracked_plates.items()
                     if current_time - plate['last_seen'] > 5.0]
        for tid in to_delete:
            del self.tracked_plates[tid]

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
        for text in texts:
            if text and text not in self.recognized_plates:
                self.recognized_plates.add(text)
                self.add_to_plates_list(text)

    def calculate_fps(self, start_time):
        """Вычисляет текущий FPS"""
        end_time = time.time()
        self.frame_times.append(end_time - start_time)
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)
        return len(self.frame_times) / sum(self.frame_times)

    def display_processed_frame(self, pil_img, video_label):
        """Отображает PIL-кадр (RGB) в интерфейсе"""
        try:
            frame_rgb = np.array(pil_img)  # уже RGB — конвертация не нужна
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            video_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Error in display_processed_frame: {e}")

    def closeEvent(self, event):
        try:
            logging.info("Closing application")
            self.release_cameras()
            logging.info("Application closed.")
        except Exception as e:
            logging.error(f"Error during close: {e}")
        event.accept()