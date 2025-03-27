import easyocr
import logging
import numpy as np
import cv2
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
    """
    Обработка изображения для обнаружения автомобилей и номерных знаков
    Возвращает:
    - Если найдены номера: [изображение с разметкой, тексты номеров, изображения номеров]
    - Если номера не найдены: [изображение с разметкой, сообщение]
    """
    license_numbers = 0
    results = {}
    licenses_texts = []
    license_plate_crops = []

    # Конвертируем изображение в BGR (для OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img

    # Детекция транспортных средств
    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    # Отрисовка bounding box'ов для автомобилей
    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
            if int(class_id) in [2, 3, 5, 7]:  # Автомобили, грузовики и т.д.
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)

    # Детекция и обработка номерных знаков
    if len(license_detections.boxes.cls.tolist()) != 0:
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Отрисовка прямоугольника вокруг номера
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Вырезаем область номера
            license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            # Распознавание текста номера
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)

            if license_plate_text:
                licenses_texts.append(license_plate_text)
                license_plate_crops.append(license_plate_crop)

                # Отрисовка текста номера
                img = draw_license_plate_text(img, license_plate_text, (int(x1), int(y1) - 40))

    # Конвертируем обратно в RGB для отображения
    img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if licenses_texts:
        return [img_wth_box, licenses_texts, license_plate_crops]
    elif len(license_detections.boxes.cls.tolist()) > 0:
        # Если были обнаружены области номеров, но текст не распознан или не соответствует формату
        return [img_wth_box, ["Распознанный текст не соответствует формату номера"]]
    else:
        # Если не обнаружено ни одной области номера
        return [img_wth_box, ["Номера не обнаружены"]]


def draw_license_plate_text(image, text, position):
    """Рисует текст номерного знака с белым фоном и черным текстом"""
    try:
        # Создаем PIL изображение из OpenCV изображения
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Загружаем шрифт DejaVuSans.ttf
        font = ImageFont.truetype("DejaVuSans.ttf", 30)

        # Рассчитываем размеры текста
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Координаты для прямоугольника фона
        x1, y1 = position
        x2 = x1 + text_width + 80
        y2 = y1 + text_height + 20

        # Рисуем белый прямоугольник фона
        draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 255))

        # Рисуем черный текст
        text_x = x1 + 40
        text_y = y1 + 10
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

        # Конвертируем обратно в OpenCV формат
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error drawing license plate text: {e}")
        return image

def license_complies_format(text):
    """Check if the license plate text complies with the required format."""
    if len(text) == 9 and text[0].isalpha() and text[1:4].isdigit() and text[4:6].isalpha() and text[6:].isdigit():
        return True
    elif len(text) == 8 and text[0].isalpha() and text[1:4].isdigit() and text[4:6].isalpha() and text[6:].isdigit():
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
    """Read the license plate text from a cropped image."""
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        # Если текст слишком короткий, пропускаем
        if len(text) < 3:
            continue

        if license_complies_format(text):
            formatted_text = format_license(text)
            formatted_text = post_process_license(formatted_text)
            return formatted_text, score
        else:
            # Возвращаем текст, даже если он не соответствует формату
            formatted_text = format_license(text)
            formatted_text = post_process_license(formatted_text)
            return f"Неизвестный формат: {formatted_text}", score

    return None, None


def post_process_license(text):
    """Post-process the recognized license plate text."""
    corrections = {
        '0': 'О', 'О': '0', 'о': '0',
        'A': 'А', 'a': 'а', 'B': 'В', 'b': 'в',
        'E': 'Е', 'e': 'е', 'K': 'К', 'k': 'к',
        'M': 'М', 'm': 'м', 'H': 'Н', 'h': 'н',
        'P': 'Р', 'p': 'р', 'C': 'С', 'c': 'с',
        'T': 'Т', 't': 'т', 'Y': 'У', 'y': 'у',
        'X': 'Х', 'x': 'х'
    }

    for char in corrections:
        text = text.replace(char, corrections[char])

    if len(text) == 9 and text[4] == '0':
        text = text[:4] + 'О' + text[5:]

    return text


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


def insert_car_data(license_plate_text, photo, car_type, date):
    """Insert car data into the database."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO car (photo, car_type, car_number, date)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (photo, car_type, license_plate_text, date))
        conn.commit()
    except mysql.connector.Error as err:
        logging.error(f"Error inserting data into database: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()