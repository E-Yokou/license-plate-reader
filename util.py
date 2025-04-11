# file util.py

import easyocr
import logging
import numpy as np
import cv2
import re
import mysql.connector
from PIL import ImageFont, ImageDraw, Image
from PyQt5.QtWidgets import QApplication

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the OCR reader
reader = easyocr.Reader(['ru'], gpu=False)

# Database configuration
db_config = {
    'host': '192.168.1.159',
    'port': 3306,
    'user': 'iwillnvrd',
    'password': 'SecurePass1212_',
    'database': 'mydatabase',
    'connect_timeout': 5
}

dict_char_to_int = {
    'О': '0', 'о': '0', 'Ы': 'М', 'Ч': 'У',
    'А': 'A', 'а': 'a', 'В': 'B', 'в': 'b',
    'Е': 'E', 'е': 'e', 'К': 'K', 'к': 'k',
    'М': 'M', 'м': 'm', 'Н': 'H', 'н': 'h',
    'Р': 'P', 'р': 'p', 'С': 'C', 'с': 'c',
    'Т': 'T', 'т': 't', 'У': 'Y', 'у': 'y',
    'Х': 'X', 'х': 'x'
}

dict_int_to_char = {v: k for k, v in dict_char_to_int.items()}


def four_point_transform(image, pts):
    """Перспективное преобразование для выравнивания номера"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # cv2.imshow('warped_image', warped)
    # cv2.waitKey(1500)
    # cv2.destroyAllWindows()

    return warped


def order_points(pts):
    """Упорядочивание точек для перспективного преобразования"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_license_plate_contour(license_plate_crop):
    """Обнаружение контура номерного знака для выравнивания"""
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None

def get_plate_center(bbox):
    """Вычисляет центр bbox номерного знака"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_car_center(bbox):
    """Вычисляет центр bbox автомобиля"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def is_plate_inside_car(plate_bbox, car_bbox):
    """Проверяет, находится ли номер внутри bbox автомобиля"""
    plate_center = get_plate_center(plate_bbox)
    xcar1, ycar1, xcar2, ycar2 = car_bbox
    return (xcar1 < plate_center[0] < xcar2 and
            ycar1 < plate_center[1] < ycar2)

def model_prediction(img, coco_model, license_plate_detector, ocr_reader, recognition_threshold=0.85):
    """Обработка изображения для обнаружения автомобилей и номерных знаков с улучшенным сопоставлением"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    licenses_texts = []
    license_plate_crops = []
    direction = None

    # Детекция транспортных средств
    object_detections = coco_model(img)[0]
    vehicle_classes = set()
    vehicle_boxes = []  # (x1, y1, x2, y2, class_id, score, area, center_x, center_y)

    for detection in object_detections.boxes.data.tolist():
        xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
        class_id = int(class_id)
        if class_id in [2, 3, 5, 7]:  # Только автомобили, грузовики и автобусы
            vehicle_classes.add(class_id)
            area = (xcar2 - xcar1) * (ycar2 - ycar1)
            center_x = (xcar1 + xcar2) / 2
            center_y = (ycar1 + ycar2) / 2
            vehicle_boxes.append((xcar1, ycar1, xcar2, ycar2, class_id, car_score, area, center_x, center_y))

            # Рисуем bounding box для транспортных средств
            cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 2)

    # Определяем направление (если нужно)
    if {2, 3, 5, 7}.intersection(vehicle_classes):
        direction = "forward"  # или другая логика определения направления

    # Детекция номерных знаков
    license_detections = license_plate_detector(img)[0]
    plate_to_car_mapping = {}

    for license_plate in license_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        plate_width = x2 - x1
        plate_height = y2 - y1
        plate_area = plate_width * plate_height
        plate_center_x = (x1 + x2) / 2
        plate_center_y = (y1 + y2) / 2

        # Рисуем прямоугольник вокруг номера
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        plate_text, plate_score = read_license_plate(license_plate_crop_gray)

        if plate_text and plate_score > recognition_threshold:
            licenses_texts.append((plate_text, plate_score))
            license_plate_crops.append(license_plate_crop)

            # Ищем лучший matching автомобиль для этого номера
            best_match = None
            best_score = 0
            min_distance = float('inf')

            for vehicle_box in vehicle_boxes:
                xcar1, ycar1, xcar2, ycar2, class_id, car_score, area, car_center_x, car_center_y = vehicle_box

                # Проверяем геометрическое соответствие (центр номера внутри bbox автомобиля)
                center_inside = (xcar1 < plate_center_x < xcar2 and
                                ycar1 < plate_center_y < ycar2)

                if not center_inside:
                    continue

                # Проверяем размерное соответствие (номер не должен быть слишком большим)
                size_ratio = plate_area / area
                if size_ratio > 0.3:  # Номер не должен занимать >30% площади авто
                    continue

                # Вычисляем расстояние между центрами
                distance = ((plate_center_x - car_center_x)**2 +
                           (plate_center_y - car_center_y)**2)**0.5

                # Score соответствия (чем меньше расстояние и size_ratio, тем лучше)
                match_score = (1 / (distance + 1)) * (1 - size_ratio) * car_score

                if match_score > best_score:
                    best_score = match_score
                    best_match = vehicle_box
                    min_distance = distance

            if best_match:
                xcar1, ycar1, xcar2, ycar2, _, _, _, _, _ = best_match
                plate_to_car_mapping[(plate_text, plate_score)] = (xcar1, ycar1, xcar2, ycar2)

    # Рисуем номера только над лучшими соответствиями
    used_vehicles = set()
    for (plate_text, plate_score), car_box in sorted(
            plate_to_car_mapping.items(),
            key=lambda x: -x[0][1]  # Сортируем по убыванию confidence номера
    ):
        if car_box in used_vehicles:
            continue  # Не рисуем несколько номеров на одном автомобиле

        xcar1, ycar1, xcar2, ycar2 = car_box
        text_position = (int(xcar1), int(ycar1) - 10)

        img = draw_license_plate_text(
            img,
            f"{plate_text} ({plate_score:.2f})",
            text_position
        )
        used_vehicles.add(car_box)

    img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if licenses_texts:
        return [img_wth_box, licenses_texts, license_plate_crops, direction]
    elif license_detections.boxes.cls.tolist():
        return [img_wth_box, [], None, direction]
    else:
        return [img_wth_box, [], None, direction]


def draw_best_result(image, best_text, best_score, position, recognition_threshold=0.85):
    """Рисует лучший результат всегда, если он есть (независимо от score)"""
    if best_text is None or best_score is None:
        return image

    try:
        # Создаем PIL изображение из OpenCV изображения
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        try:
            # Пробуем загрузить DejaVuSans.ttf, если не получится - используем стандартный шрифт
            font = ImageFont.truetype("DejaVuSans.ttf", 30)
        except:
            font = ImageFont.load_default()

        text = f"{best_text}"

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x1, y1 = position
        x2 = x1 + text_width + 20
        y2 = y1 + text_height + 10

        # Белый фон с чёрным текстом
        draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 255), outline=None, width=1)
        draw.text((x1 + 10, y1 + 5), text, font=font, fill=(0, 0, 0))

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    except Exception as e:
        logging.error(f"Error drawing best result: {e}")
        return image

def draw_license_plate_text(image, text, position, font_size=30, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """Рисует текст номерного знака с фоном"""
    try:
        # Создаем PIL изображение из OpenCV изображения
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        try:
            # Пробуем загрузить DejaVuSans.ttf
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Рассчитываем размеры текста
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Координаты для прямоугольника фона
        x1, y1 = position
        x2 = x1 + text_width + 20
        y2 = y1 + text_height + 10

        # Рисуем прямоугольник фона
        draw.rectangle([(x1, y1), (x2, y2)], fill=bg_color)

        # Рисуем текст
        text_x = x1 + 10
        text_y = y1 + 5
        draw.text((text_x, text_y), text, font=font, fill=text_color)

        # Конвертируем обратно в OpenCV формат
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error drawing license plate text: {e}")
        return image


def license_complies_format(text, recognition_threshold=0.85):
    """Проверка формата номера (Российский стандарт)"""
    # Если обработка под шаблоны отключена, всегда возвращаем True
    if not getattr(QApplication.instance(), 'template_processing_enabled', True):
        return True

    # car
    if len(text) == 9 and text[0].isalpha() and text[1:4].isdigit() and text[4:6].isalpha() and text[6:].isdigit():
        return True
    # car
    elif len(text) == 8 and text[0].isalpha() and text[1:4].isdigit() and text[4:6].isalpha() and text[6:].isdigit():
        return True
    # trailer
    elif len(text) == 8 and text[0:2].isalpha() and text[2:6].isdigit() and text[6:].isdigit():
        return True
    # trailer
    elif len(text) == 9 and text[0:2].isalpha() and text[2:6].isdigit() and text[6:].isdigit():
        return True
    return False


def format_license(text):
    """Format the license plate text by converting characters."""
    license_plate_ = ''
    for char in text:
        if char in dict_char_to_int:
            license_plate_ += dict_char_to_int[char]
        elif char in dict_int_to_char:
            license_plate_ += dict_int_to_char[char]
        else:
            license_plate_ += char
    return license_plate_


def read_license_plate(license_plate_crop):
    """Улучшенное распознавание номера с проверкой формата"""
    try:
        # Предварительная обработка изображения
        processed_img = preprocess_image(license_plate_crop)

        # Распознавание текста
        detections = reader.readtext(
            processed_img,
            decoder='greedy',
            batch_size=1,
            detail=1,
            # allowlist='АВЕКМНОРСТУХ0123456789',
            paragraph=False
        )

        best_text = None
        best_score = 0

        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '').replace('-', '')

            # Удаляем лишние символы
            text = ''.join(c for c in text if c.isalnum())

            if len(text) < 5:  # Минимальная длина номера
                continue

            # Проверка формата номера
            formatted_text = format_license(text)
            if not license_complies_format(formatted_text):
                continue

            if score > best_score:
                best_text = formatted_text
                best_score = score

        return best_text, best_score if best_text else (None, 0.0)

    except Exception as e:
        logging.error(f"Error in read_license_plate: {e}")
        return None, 0.0


def draw_tracking_info(image, car_bbox, plate_text=None, plate_score=None):
    """Рисует информацию о трекинге на изображении"""
    x1, y1, x2, y2 = map(int, car_bbox)

    # Рисуем bounding box автомобиля
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Если есть номер, рисуем его
    if plate_text:
        text = f"{plate_text} ({plate_score:.2f})" if plate_score else plate_text
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Фон для текста
        cv2.rectangle(image,
                      (x1, y1 - text_height - 10),
                      (x1 + text_width + 10, y1),
                      (0, 0, 255), -1)

        # Сам текст
        cv2.putText(image, text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    return image

def draw_tracked_plate(image, text, car_bbox, score=0.0):
    """Рисует номер над автомобилем с bounding box"""
    try:
        x1, y1, x2, y2 = car_bbox

        # Рисуем bounding box автомобиля
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Создаем PIL изображение для текста
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()

        display_text = f"{text} ({score:.2f})"

        # Рассчитываем размер текста
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Позиция над bounding box
        text_x = int((x1 + x2) / 2 - text_width / 2)
        text_y = y1 - text_height - 10

        # Рисуем прямоугольник фона
        draw.rectangle(
            [(text_x - 5, text_y - 5),
             (text_x + text_width + 5, text_y + text_height + 5)],
            fill=(0, 0, 255)  # Синий фон
        )

        # Рисуем текст
        draw.text(
            (text_x, text_y),
            display_text,
            font=font,
            fill=(255, 255, 255)  # Белый текст
        )

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error drawing tracked plate: {e}")
        return image

def preprocess_image(img):
    """Улучшенная предобработка изображения"""
    try:
        return img
    except Exception as e:
        logging.error(f"Error in preprocess_image: {e}")
        return img


def post_process_license(text):
    """Исправление частых ошибок распознавания"""
    corrections = {
        'И': 'Н', 'П': 'Н', 'Л': 'Е', 'Ц': '7',
        'Ч': '9', 'Я': '9', 'З': '3', 'Ш': 'Н',
        'Ъ': 'Ь', 'Ы': 'М', 'В': 'В', 'Д': '0',
        ' ': '', '-': '', '_': ''
    }

    # Применяем замену символов
    corrected_text = []
    for char in text.upper():
        corrected_text.append(corrections.get(char, char))

    return ''.join(corrected_text)


def get_car(license_plate, vehicle_track_ids):
    """Match a license plate to a vehicle in the tracked vehicles list."""
    x1, y1, x2, y2, score, class_id = license_plate
    foundIt = False

    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]
    return -1, -1, -1, -1, -1


def insert_car_data(license_plate_text, photo, car_type, date, camera_id):
    """Insert car data into the database with duplicate check."""
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # First, check if this plate was already recorded recently (last 5 minutes)
        check_query = """
        SELECT id FROM car 
        WHERE car_number = %s AND date >= DATE_SUB(%s, INTERVAL 1 MINUTE)
        LIMIT 1
        """
        cursor.execute(check_query, (license_plate_text, date))
        if cursor.fetchone():
            logging.info(f"Plate {license_plate_text} already recorded recently - skipping")
            return

        # Get camera ID if camera name was provided
        if camera_id and not str(camera_id).isdigit():
            cursor.execute("SELECT id FROM camera WHERE name = %s", (camera_id,))
            result = cursor.fetchone()
            if result:
                camera_id = result[0]
            else:
                camera_id = None

        insert_query = """
        INSERT INTO car (photo, car_type, car_number, date, camera_id)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (photo, car_type, license_plate_text, date, camera_id))
        conn.commit()
        logging.info(f"Successfully inserted plate: {license_plate_text}")

    except mysql.connector.Error as err:
        logging.error(f"MySQL error: {err}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()