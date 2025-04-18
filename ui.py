# ui.py
import base64
import logging
import queue
from datetime import datetime
import threading
import time

import mysql
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QWidget, QLabel, QSizePolicy, QTextEdit, QPushButton, QVBoxLayout, QApplication,
    QMenuBar, QMenu, QAction, QInputDialog, QLineEdit, QCheckBox, QDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QFileDialog, QMessageBox, QComboBox, QTableWidgetItem, QDialogButtonBox,
    QTableWidget, QListWidgetItem, QListWidget, QGridLayout, QSlider, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QPen
from PIL import ImageFont, ImageDraw, Image
import cv2
import requests
from util import model_prediction, reader, draw_best_result, db_config, draw_tracked_plate, license_complies_format, \
    read_license_plate, get_plate_center, get_car_center, is_plate_inside_car, draw_tracking_info
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


class StreamingSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки трансляции")
        self.parent = parent

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Группа настроек частоты
        freq_group = QGroupBox("Частота отправки")
        freq_layout = QFormLayout()

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.01, 5.0)
        self.interval_spin.setSingleStep(0.05)
        self.interval_spin.setSuffix(" сек")
        freq_layout.addRow("Интервал между кадрами:", self.interval_spin)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 10)
        freq_layout.addRow("Количество потоков:", self.threads_spin)

        freq_group.setLayout(freq_layout)
        self.layout.addWidget(freq_group)

        # Группа настроек качества
        quality_group = QGroupBox("Качество изображения")
        quality_layout = QFormLayout()

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080", "Исходное"])
        quality_layout.addRow("Разрешение:", self.resolution_combo)

        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(30, 100)
        self.quality_slider.setTickInterval(10)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        quality_layout.addRow("Качество JPEG (%):", self.quality_slider)

        quality_group.setLayout(quality_layout)
        self.layout.addWidget(quality_group)

        # Кнопки
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def set_values(self, interval, threads, resolution, quality):
        self.interval_spin.setValue(interval)
        self.threads_spin.setValue(threads)

        index = self.resolution_combo.findText(resolution)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)

        self.quality_slider.setValue(quality)

    def get_values(self):
        return {
            'interval': self.interval_spin.value(),
            'threads': self.threads_spin.value(),
            'resolution': self.resolution_combo.currentText(),
            'quality': self.quality_slider.value()
        }

class CameraSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор камер")
        self.setMinimumSize(400, 300)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.camera_list = QListWidget(self)
        self.layout.addWidget(self.camera_list)

        self.select_button = QPushButton("Выбрать", self)
        self.select_button.clicked.connect(self.accept)
        self.layout.addWidget(self.select_button)

        self.load_cameras()

    def load_cameras(self):
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, ip_address, name FROM camera")
            cameras = cursor.fetchall()
            cursor.close()
            conn.close()

            for camera in cameras:
                item = QListWidgetItem(f"{camera['name']} ({camera['ip_address']})")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, camera['ip_address'])
                self.camera_list.addItem(item)

        except mysql.connector.Error as err:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке камер: {err}")

    def get_selected_cameras(self):
        selected_cameras = []
        for index in range(self.camera_list.count()):
            item = self.camera_list.item(index)
            if item.checkState() == Qt.Checked:
                selected_cameras.append(item.data(Qt.UserRole))
        return selected_cameras

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


class CameraManagementDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Управление камерами")
        self.setMinimumSize(600, 400)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Кнопка добавления новой камеры
        self.add_button = QPushButton("Добавить новую камеру")
        self.add_button.clicked.connect(self.add_camera)
        self.layout.addWidget(self.add_button)

        # Таблица с камерами
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "IP адрес", "Название", "Действия"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.layout.addWidget(self.table)

        # Кнопки управления
        self.button_box = QDialogButtonBox(QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        # Загружаем данные камер
        self.load_cameras()

    def load_cameras(self):
        """Загружает список камер из базы данных"""
        try:
            # Подключаемся к базе данных
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            # Получаем все камеры
            cursor.execute("SELECT id, ip_address, name FROM camera")
            cameras = cursor.fetchall()

            # Заполняем таблицу
            self.table.setRowCount(len(cameras))
            for row, camera in enumerate(cameras):
                # ID
                id_item = QTableWidgetItem(str(camera['id']))
                id_item.setFlags(id_item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(row, 0, id_item)

                # IP адрес
                ip_item = QTableWidgetItem(camera['ip_address'])
                self.table.setItem(row, 1, ip_item)

                # Название
                name_item = QTableWidgetItem(camera['name'])
                self.table.setItem(row, 2, name_item)

                # Кнопки действий
                button_widget = QWidget()
                button_layout = QHBoxLayout()
                button_layout.setContentsMargins(0, 0, 0, 0)

                update_btn = QPushButton("Изменить")
                update_btn.clicked.connect(lambda _, r=row: self.update_camera(r))
                button_layout.addWidget(update_btn)

                delete_btn = QPushButton("Удалить")
                delete_btn.clicked.connect(lambda _, r=row: self.delete_camera(r))
                button_layout.addWidget(delete_btn)

                button_widget.setLayout(button_layout)
                self.table.setCellWidget(row, 3, button_widget)

            cursor.close()
            conn.close()

        except mysql.connector.Error as err:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке камер: {err}")

    def add_camera(self):
        """Добавляет новую камеру"""
        try:
            # Получаем данные из диалога
            ip, ok = QInputDialog.getText(self, "Добавить камеру", "Введите IP адрес камеры:")
            if not ok or not ip:
                return

            name, ok = QInputDialog.getText(self, "Добавить камеру", "Введите название камеры:")
            if not ok:
                return

            # Сохраняем в базу данных
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO camera (ip_address, name) VALUES (%s, %s)",
                (ip, name)
            )
            conn.commit()

            cursor.close()
            conn.close()

            # Обновляем таблицу
            self.load_cameras()

        except mysql.connector.Error as err:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при добавлении камеры: {err}")

    def update_camera(self, row):
        """Обновляет данные камеры"""
        try:
            # Получаем данные из таблицы
            camera_id = int(self.table.item(row, 0).text())
            ip_address = self.table.item(row, 1).text()
            name = self.table.item(row, 2).text()

            # Обновляем в базе данных
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE camera SET ip_address = %s, name = %s WHERE id = %s",
                (ip_address, name, camera_id)
            )
            conn.commit()

            cursor.close()
            conn.close()

            QMessageBox.information(self, "Успех", "Данные камеры обновлены")

        except mysql.connector.Error as err:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обновлении камеры: {err}")

    def delete_camera(self, row):
        """Удаляет камеру"""
        try:
            camera_id = int(self.table.item(row, 0).text())

            # Подтверждение удаления
            reply = QMessageBox.question(
                self, 'Подтверждение',
                'Вы уверены, что хотите удалить эту камеру?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

            # Удаляем из базы данных
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM camera WHERE id = %s", (camera_id,))
            conn.commit()

            cursor.close()
            conn.close()

            # Обновляем таблицу
            self.load_cameras()

        except mysql.connector.Error as err:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при удалении камеры: {err}")

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
        self.current_camera_url = None
        self.camera_ids = []  # Список идентификаторов камер

        self.camera_info = {}  # {camera_id: {'ip': str, 'name': str}}
        self.current_camera_id = None  # Текущий активный camera_id

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

        # Настроки трансляции
        self.streaming_interval = 0.1
        self.last_streaming_time = 0
        self.streaming_threads = []
        self.streaming_queue = queue.Queue(maxsize=10)

        self.streaming_settings = {
            'interval': 0.1,
            'threads': 3,
            'resolution': '1280x720',
            'quality': 70
        }

        self.tracked_plates = {}
        self.tracked_plates = {}  # {track_id: {'plate_text': str, 'plate_score': float, 'last_seen': float}}
        self.plate_history = {}  # Для хранения истории номеров

        # Добавляем список для хранения названий камер
        self.camera_names = []

        # Connect signals
        self.connection_status_changed.connect(self.update_connection_status)
        self.camera_connected_changed.connect(self.update_camera_button_status)

    def init_ui(self):
        self.setWindowTitle("Vehicle & License Plate Recognition")
        self.setGeometry(100, 100, 800, 600)

        # Create main widgets
        self.video_labels = []
        self.video_layout = QGridLayout()

        # Create bottom panel with minimal buttons
        self.bottom_panel = QHBoxLayout()

        self.pause_button = QPushButton("Пауза", self)
        self.pause_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        self.bottom_panel.addWidget(self.pause_button)

        self.connect_button = QPushButton("Подключить", self)
        self.connect_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.connect_button.clicked.connect(self.connect_cameras)
        self.bottom_panel.addWidget(self.connect_button)

        # Add camera switch button
        self.camera_switch_button = QPushButton("Переключить камеру", self)
        self.camera_switch_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.camera_switch_button.clicked.connect(self.switch_camera)
        self.camera_switch_button.setEnabled(False)
        self.bottom_panel.addWidget(self.camera_switch_button)

        # Create main layout
        layout = QVBoxLayout()
        layout.addLayout(self.video_layout, 8)
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

        self.camera_management_action = QAction("Управление камерами", self)
        self.camera_management_action.triggered.connect(self.show_camera_management)
        self.camera_menu.addAction(self.camera_management_action)

        self.select_camera_action = QAction("Выбрать камеру", self)
        self.select_camera_action.triggered.connect(self.select_camera)
        self.camera_menu.addAction(self.select_camera_action)
        self.menu_bar.addMenu(self.camera_menu)

        # Stream menu
        self.stream_menu = QMenu("Трансляция", self.menu_bar)

        self.stream_settings_action = QAction("Настройки сервера", self)
        self.stream_settings_action.triggered.connect(self.show_server_settings)
        self.stream_menu.addAction(self.stream_settings_action)

        self.stream_params_action = QAction("Параметры трансляции", self)
        self.stream_params_action.triggered.connect(self.show_streaming_settings)
        self.stream_menu.addAction(self.stream_params_action)

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

    def prepare_frame_for_streaming(self, frame):
        """Подготавливает кадр для отправки согласно настройкам"""
        try:
            # Изменяем разрешение
            if self.streaming_settings['resolution'] != "Исходное":
                width, height = map(int, self.streaming_settings['resolution'].split('x'))
                frame = cv2.resize(frame, (width, height))

            # Кодируем в JPEG с заданным качеством
            quality = self.streaming_settings['quality']
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            if not success:
                logging.error("Failed to encode frame to JPEG")
                return None

            return buffer

        except Exception as e:
            logging.error(f"Error preparing frame: {e}")
            return None

    def show_streaming_settings(self):
        try:
            dialog = StreamingSettingsDialog(self)
            dialog.set_values(
                self.streaming_settings['interval'],
                self.streaming_settings['threads'],
                self.streaming_settings['resolution'],
                self.streaming_settings['quality']
            )

            if dialog.exec_():
                new_settings = dialog.get_values()
                self.streaming_settings.update(new_settings)

                # Применяем новые настройки
                self.start_streaming_threads(self.streaming_settings['threads'])

                logging.info(f"Streaming settings updated: {self.streaming_settings}")

        except Exception as e:
            logging.error(f"Error in streaming settings dialog: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка в настройках трансляции: {str(e)}")

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

    def show_camera_selection(self):
        try:
            logging.info("Opening camera selection dialog")
            dialog = CameraSelectionDialog(self)
            if dialog.exec_():
                selected_cameras = dialog.get_selected_cameras()
                logging.info(f"Selected cameras: {selected_cameras}")
                self.connect_selected_cameras(selected_cameras)
        except Exception as e:
            logging.error(f"Error in camera selection dialog: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при выборе камер: {str(e)}")

    def connect_selected_cameras(self, selected_cameras):
        try:
            logging.info("Connecting to selected cameras")
            self.release_cameras()
            self.video_labels.clear()
            self.camera_ids = []
            self.camera_info.clear()  # Очищаем информацию о камерах

            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, ip_address, name FROM camera WHERE ip_address IN (%s)" % ','.join(
                ['%s'] * len(selected_cameras)), selected_cameras)
            cameras = cursor.fetchall()
            cursor.close()
            conn.close()

            for row, camera in enumerate(cameras):
                cap = cv2.VideoCapture(camera['ip_address'])
                if not cap.isOpened():
                    logging.error(f"Error opening video stream for camera: {camera['ip_address']}")
                    continue

                video_label = QLabel(self)
                video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                video_label.setAlignment(Qt.AlignCenter)
                self.video_labels.append((cap, video_label))
                self.camera_ids.append(camera['id'])

                # Сохраняем информацию о камере
                self.camera_info[camera['id']] = {
                    'ip': camera['ip_address'],
                    'name': camera['name']
                }

                self.video_layout.addWidget(video_label, row // 2, row % 2)
                self.camera_names.append(camera['name'])

            self.timer.start(1000 // self.fps_limit)
            self.connect_button.setText("Отключить")
            self.pause_button.setEnabled(True)
            self.camera_connected_changed.emit(True)
            logging.info("Cameras connected")
        except Exception as e:
            logging.error(f"Error connecting to selected cameras: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при подключении к выбранным камерам: {str(e)}")

    def show_camera_settings(self):
        try:
            logging.info("Opening camera management dialog")
            dialog = CameraManagementDialog(self)
            dialog.exec_()
        except Exception as e:
            logging.error(f"Error in camera management dialog: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка в управлении камерами: {str(e)}")

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

    def get_cameras_from_db(self):
        """Получает список камер из базы данных"""
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT id, ip_address, name FROM camera")
            cameras = cursor.fetchall()

            cursor.close()
            conn.close()

            return cameras

        except mysql.connector.Error as err:
            logging.error(f"Database error in get_cameras_from_db: {err}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке камер из БД: {str(err)}")
            return []

    def select_camera(self):
        try:
            logging.info("Selecting camera")

            # Получаем список камер из базы данных
            cameras = self.get_cameras_from_db()

            # Собираем все доступные варианты камер
            items = []
            camera_types = []

            # Добавляем камеры из базы данных
            for camera in cameras:
                display_text = f"{camera['name']} ({camera['ip_address']})"
                items.append(display_text)
                camera_types.append(("db_camera", camera['ip_address']))

            # Добавляем остальные камеры
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

            if not items:
                QMessageBox.warning(self, "Ошибка",
                                    "Нет доступных камер. Настройте камеры в меню 'Управление камерами'")
                return

            item, ok = QInputDialog.getItem(
                self,
                "Выбор камеры",
                "Выберите камеру:",
                items,
                0,
                False
            )

            if ok and item:
                selected_index = items.index(item)
                camera_type, camera_param = camera_types[selected_index]

                if camera_type == "db_camera":
                    self.current_camera = f"Камера из БД ({camera_param})"
                    self.current_camera_url = camera_param
                elif camera_type == "static_camera":
                    self.current_camera = f"Камера {camera_param + 1}"
                    self.current_camera_url = self.camera_urls[camera_param]
                elif camera_type == "usb_camera":
                    self.current_camera = "USB Камера"
                    self.usb_camera_index = camera_param
                elif camera_type == "video_file":
                    self.current_camera = "Видеофайл"

                logging.info(f"Selected camera: {self.current_camera}")
                self.update_camera_buttons()

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
                # Disconnect cameras
                self.timer.stop()
                self.release_cameras()
                self.connect_button.setText("Подключить")
                self.pause_button.setEnabled(False)
                self.camera_connected_changed.emit(False)
                self.clear_video_streams()
                logging.info("Cameras disconnected")
            else:
                try:
                    # Попытка подключения к базе данных
                    conn = mysql.connector.connect(**db_config)
                    conn.close()
                    # Если подключение успешно, показываем диалог выбора камер
                    self.show_camera_selection()
                except mysql.connector.Error as err:
                    # Если не удалось подключиться к MySQL, предлагаем ручной ввод
                    reply = QMessageBox.question(
                        self, 'Ошибка подключения',
                        'Не удалось подключиться к серверу MySQL. Хотите ввести URL камер вручную?',
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                    )
                    if reply == QMessageBox.Yes:
                        self.add_cameras_manually()
                if self.current_camera == "Видеофайл":
                    self.load_video()
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
            self.camera_ids = []
            self.camera_info = {}
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
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    video_label = QLabel(self)
                    video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    video_label.setAlignment(Qt.AlignCenter)
                    self.video_labels.append((cap, video_label))
                    self.video_layout.addWidget(video_label, i // 2, i % 2)

                    self.camera_ids.append(i)  # Используем индекс как ID
                    self.camera_names.append(f"Ручная камера {i + 1}")

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

    def restart_video_streams(self):
        self.release_cameras()

        try:
            if self.current_camera.startswith("Камера из БД") and hasattr(self, 'current_camera_url'):
                # Обработка камеры из базы данных
                self.cap1 = cv2.VideoCapture(self.current_camera_url)
                if not self.cap1.isOpened():
                    logging.error(f"Error opening video stream from DB: {self.current_camera_url}")
                    self.cap1 = None
                    return False
                logging.info(f"Successfully connected to camera from DB: {self.current_camera_url}")
                return True

            elif self.current_camera == "Камера 1" and self.camera_urls[0]:
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
            self.start_streaming_threads()  # Запускаем потоки
            logging.info("Трансляция на сервер запущена")
            self.connection_status_changed.emit("connected")

    def stop_streaming(self):
        if self.is_streaming:
            self.is_streaming = False
            self.stop_streaming_threads()  # Останавливаем потоки
            logging.info("Трансляция на сервер остановлена")
            if self.connection_status == "connected":
                self.connection_status_changed.emit("disconnected")

    def set_streaming_settings(self):
        """Настройка параметров трансляции"""
        try:
            interval, ok = QInputDialog.getDouble(
                self, "Интервал трансляции",
                "Введите интервал между кадрами (секунды):",
                self.streaming_interval, 0.01, 1.0, 2
            )
            if ok:
                self.streaming_interval = interval

            threads, ok = QInputDialog.getInt(
                self, "Количество потоков",
                "Введите количество потоков для отправки:",
                len(self.streaming_threads), 1, 10, 1
            )
            if ok:
                self.start_streaming_threads(threads)

        except Exception as e:
            logging.error(f"Error setting streaming params: {e}")

    def send_frame(self, frame, camera_id):
        """Отправляет кадр на сервер трансляции"""
        try:
            if not self.server_ip or not self.server_port:
                logging.error("Server IP or port not set")
                return False

            # Подготавливаем кадр
            buffer = self.prepare_frame_for_streaming(frame)
            if buffer is None:
                return False

            # Формируем payload
            payload = {
                "camera_id": str(camera_id),
                "timestamp": datetime.now().isoformat(),
                "frame": base64.b64encode(buffer).decode('utf-8'),
                "settings": self.streaming_settings
            }

            # Отправка на сервер
            response = requests.post(
                f"http://{self.server_ip}:{self.server_port}/upload",
                json=payload,
                timeout=1.0
            )

            if response.status_code != 200:
                logging.error(f"Server error: {response.status_code} - {response.text}")
                self.connection_status_changed.emit("disconnected")
                return False

            return True

        except Exception as e:
            logging.error(f"Error in send_frame: {e}")
            self.connection_status_changed.emit("disconnected")
            return False

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

    def show_camera_management(self):
        """Показывает диалог управления камерами"""
        try:
            logging.info("Opening camera management dialog")
            dialog = CameraManagementDialog(self)
            dialog.exec_()

            self.update_camera_list_from_db()
        except Exception as e:
            logging.error(f"Error in camera management dialog: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка в управлении камерами: {str(e)}")

    def update_camera_list_from_db(self):
        """Обновляет список камер из базы данных"""
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT ip_address FROM camera")
            cameras = cursor.fetchall()

            for i, camera in enumerate(cameras[:2]):
                self.camera_urls[i] = camera['ip_address']

            cursor.close()
            conn.close()

        except mysql.connector.Error as err:
            logging.error(f"Database error: {err}")

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

                # Получаем ID и название камеры
                camera_id = self.camera_ids[idx] if idx < len(self.camera_ids) else idx
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

                            # Сохранение в базу данных
                            if plate_text != getattr(self, f'last_saved_plate_{camera_id}', None):
                                try:
                                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
                                    self.insert_car_data(
                                        plate_text,
                                        buffer.tobytes(),
                                        "Car",
                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        camera_id
                                    )
                                    setattr(self, f'last_saved_plate_{camera_id}', plate_text)
                                except Exception as e:
                                    logging.error(f"Ошибка сохранения в БД: {e}")

                # 3. Визуализация номеров на автомобилях
                for track in track_ids:
                    x1, y1, x2, y2, track_id = track
                    plate_info = self.tracked_plates.get(track_id)

                    if plate_info:
                        text = f"{plate_info['plate_text']} ({plate_info['plate_score']:.2f})"
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

                # Конвертируем обратно в OpenCV формат
                processed_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                self.display_processed_frame(processed_frame, video_label)

                # Отправка на сервер при необходимости
                if self.is_streaming:
                    self.send_frame_to_stream(processed_frame, idx)

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

    def clean_old_tracks(self):
        """Очищает треки, которые не обновлялись более 5 секунд"""
        current_time = time.time()
        to_delete = [tid for tid, plate in self.tracked_plates.items()
                     if current_time - plate['last_seen'] > 5.0]
        for tid in to_delete:
            del self.tracked_plates[tid]

    def save_to_database(self, plate_text, frame, car_type, camera_id=None):
        """Сохраняет данные в базу данных"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            if buffer is not None:
                photo_bytes = buffer.tobytes()
                current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.insert_car_data(
                    plate_text,
                    photo_bytes,
                    car_type,
                    current_date,
                    camera_id
                )
        except Exception as e:
            logging.error(f"Ошибка при сохранении в БД: {e}")

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

    def display_processed_frame(self, frame, video_label):
        """Отображает обработанный кадр в интерфейсе"""
        try:
            # Получаем индекс камеры
            idx = next(i for i, (_, label) in enumerate(self.video_labels) if label == video_label)

            # Получаем название камеры
            camera_name = self.camera_names[idx] if idx < len(self.camera_names) else f"Камера {idx + 1}"

            # Создаем изображение с текстом названия камеры
            height, width = frame.shape[:2]

            # Конвертируем в PIL для использования DejaVuSans
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 30)
            except:
                font = ImageFont.load_default()

            text = camera_name
            margin = 10

            # Рассчитываем размер текста
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Позиция в нижнем левом углу
            pos_x = margin
            pos_y = height - margin - text_height

            # Рисуем красный прямоугольник
            draw.rectangle(
                [(pos_x, pos_y), (pos_x + text_width + 20, pos_y + text_height + 10)],
                fill=(255, 0, 0)  # Красный
            )

            # Рисуем белый текст
            draw.text(
                (pos_x + 10, pos_y + 5),
                text,
                font=font,
                fill=(255, 255, 255)  # Белый
            )

            # Конвертируем обратно в OpenCV формат
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Отображаем кадр
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            video_label.setPixmap(pixmap)

        except Exception as e:
            logging.error(f"Error in display_processed_frame: {e}")
            # Fallback to basic display if error occurs
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            video_label.setPixmap(pixmap)

    def send_frame_to_stream(self, frame, camera_idx):
        """Добавляет кадр в очередь для отправки на сервер"""
        try:
            current_time = time.time()
            last_time = getattr(self, f'last_send_time_{camera_idx}', 0)

            # Проверяем интервал для этой камеры
            if current_time - last_time < self.streaming_settings['interval']:
                return  # Пропускаем кадр, если не прошло достаточно времени

            # Используем camera_idx напрямую, если camera_ids не заполнен
            camera_id = camera_idx
            if camera_idx < len(self.camera_ids):
                camera_id = self.camera_ids[camera_idx]

            # Добавляем в очередь, если есть место
            try:
                self.streaming_queue.put_nowait((frame.copy(), camera_id))
                setattr(self, f'last_send_time_{camera_idx}', current_time)
            except queue.Full:
                pass

        except Exception as e:
            logging.error(f"Error in send_frame_to_stream: {e}")

    def streaming_worker(self):
        """Рабочий поток для отправки кадров на сервер"""
        while True:
            try:
                frame, camera_id = self.streaming_queue.get()
                if frame is None:  # Сигнал остановки
                    break

                # Отправляем только если трансляция активна
                if self.is_streaming:
                    self.send_frame(frame, camera_id)

            except Exception as e:
                logging.error(f"Error in streaming worker: {e}")
                time.sleep(0.1)  # задержка при ошибках

    def start_streaming_threads(self, num_threads=3):
        """Запускает несколько потоков для отправки кадров"""
        self.stop_streaming_threads()  # Останавливаем существующие потоки

        for _ in range(num_threads):
            thread = threading.Thread(target=self.streaming_worker, daemon=True)
            thread.start()
            self.streaming_threads.append(thread)

    def stop_streaming_threads(self):
        """Останавливает все потоки отправки"""
        for _ in range(len(self.streaming_threads)):
            self.streaming_queue.put(None)  # Отправляем сигнал остановки

        for thread in self.streaming_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

        self.streaming_threads = []

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