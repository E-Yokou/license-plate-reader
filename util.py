# file util.py

import easyocr
import logging
import numpy as np
import cv2
import re
import mysql.connector
from PIL import ImageFont, ImageDraw, Image

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the OCR reader
reader = easyocr.Reader(['ru'], gpu=False)

# Database configuration
db_config = {
    'host': '192.168.1.159',
    'port': 3306,
    'user': 'iwillnvrd',
    'password': 'SecurePass1212_',
    'database': 'mydatabase'
}

dict_char_to_int = {
    'О': '0', 'о': '0',
    'А': 'A', 'а': 'a', 'В': 'B', 'в': 'b',
    'Е': 'E', 'е': 'e', 'К': 'K', 'к': 'k',
    'М': 'M', 'м': 'm', 'Н': 'H', 'н': 'h',
    'Р': 'P', 'р': 'p', 'С': 'C', 'с': 'c',
    'Т': 'T', 'т': 't', 'У': 'Y', 'у': 'y',
    'Х': 'X', 'х': 'x'
}

dict_int_to_char = {v: k for k, v in dict_char_to_int.items()}


def model_prediction(img, coco_model, license_plate_detector, ocr_reader):
    """Обработка изображения для обнаружения автомобилей и номерных знаков"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    licenses_texts = []
    license_plate_crops = []
    plates_to_draw = []
    direction = None

    # Детекция транспортных средств
    object_detections = coco_model(img)[0]
    vehicle_classes = set()  # Храним все обнаруженные классы

    for detection in object_detections.boxes.data.tolist():
        xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
        class_name = coco_model.names[int(class_id)]
        vehicle_classes.add(int(class_id))

        # Рисуем bounding box для транспортных средств
        if int(class_id) in [2, 3, 5, 7]:  # Транспортные средства
            cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
        elif int(class_id) in [72, 73]:  # Холодильник или поезд
            cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (255, 0, 0), 3)

    # Определяем направление
    direction = ""
    if {6, 72}.intersection(vehicle_classes):
        direction = "backward"
    elif {7}.intersection(vehicle_classes):  # Если есть транспортное средство
        direction = "forward"

    # Детекция номерных знаков
    license_detections = license_plate_detector(img)[0]
    for license_plate in license_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Рисуем прямоугольник вокруг номерного знака (зеленый)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        plate_text, plate_score = read_license_plate(license_plate_crop_gray)

        if plate_text and plate_score > 0.85:  # Фильтрация по score
            # Создаем PIL изображение для отрисовки текста с DejaVuSans
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 30)
            except:
                font = ImageFont.load_default()

            text = f"{plate_text} ({plate_score:.2f})"

            # Рассчитываем размер текста
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Позиция текста над bounding box'ом номера
            text_x = int(x1)
            text_y = int(y1) - text_height - 10 if int(y1) - text_height - 10 > 0 else int(y1) + 10

            # Рисуем прямоугольник фона (белый)
            draw.rectangle(
                [(text_x, text_y), (text_x + text_width + 20, text_y + text_height + 10)],
                fill=(255, 255, 255)
            )

            # Рисуем текст (черный)
            draw.text(
                (text_x + 10, text_y + 5),
                text,
                font=font,
                fill=(0, 0, 0)
            )

            # Конвертируем обратно в OpenCV формат
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            plates_to_draw.append({
                'text': plate_text,
                'score': plate_score,
                'position': (int(x1), int(y1) - 40),
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            })

    # Находим номер с максимальным score в текущем кадре
    current_best = max(plates_to_draw, key=lambda x: x['score'], default=None)

    img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if plates_to_draw:
        return [img_wth_box, licenses_texts, license_plate_crops,
                current_best['text'], current_best['score'], direction]
    elif license_detections.boxes.cls.tolist():
        return [img_wth_box, ["Номера обнаружены, но score < 0.85"], None, None, 0, direction]
    else:
        return [img_wth_box, ["Номера не обнаружены"], None, None, 0, direction]


def draw_best_result(image, best_text, best_score, position):
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
        draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 255))
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


def license_complies_format(text):
    """Проверка формата номера (Российский стандарт)"""
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
    """Read the license plate text from a cropped image with enhanced preprocessing."""
    try:
        # Предварительная обработка изображения
        processed_img = preprocess_image(license_plate_crop)

        # cv2.resize(processed_img, (200, 50))

        # Параметры для улучшения распознавания
        decoder = 'greedy'  # Можно попробовать 'beamsearch'
        beamWidth = 5  # Для beamsearch
        batch_size = 1

        detections = reader.readtext(
            processed_img,
            decoder=decoder,
            beamWidth=beamWidth,
            batch_size=batch_size,
            detail=1,
            paragraph=False,
            min_size=20,
            text_threshold=0.7,
            low_text=0.4,
            link_threshold=0.4,
            canvas_size=2560,
            mag_ratio=1.5
        )

        best_text = None
        best_score = 0

        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '').replace('-', '')

            # Удаляем лишние символы
            text = ''.join(c for c in text if c.isalnum())

            if len(text) < 3:
                continue

            # Проверяем формат номера
            formatted_text = format_license(text)
            formatted_text = post_process_license(formatted_text)

            # Сохраняем результат с наивысшим score
            if score > best_score:
                best_text = formatted_text
                best_score = score

        if best_text:
            if license_complies_format(best_text):
                logging.info(f"Valid plate: {best_text}, score: {best_score:.2f}")
                return best_text, float(best_score)  # Явно преобразуем в float
            else:
                logging.info(f"Invalid format: {best_text}, score: {best_score:.2f}")
        else:
            logging.info("No plate found")

        return None, 0.0  # Всегда возвращаем кортеж (None, 0.0) если номер не найден

    except Exception as e:
        logging.error(f"Error in read_license_plate: {e}")
        return None, 0.0


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
    """Insert car data into the database."""
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # First, get the camera ID if camera name was provided
        if camera_id and not str(camera_id).isdigit():
            # Query the database to get camera ID by name
            cursor.execute("SELECT id FROM camera WHERE name = %s", (camera_id,))
            result = cursor.fetchone()
            if result:
                camera_id = result[0]
            else:
                camera_id = None  # or set a default value

        insert_query = """
        INSERT INTO car (photo, car_type, car_number, date, camera_id)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (photo, car_type, license_plate_text, date, camera_id))
        conn.commit()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка MySQL: {err}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()