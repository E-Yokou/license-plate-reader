## @file util.py
# @brief Utility functions for license plate recognition.

import easyocr

# Initialize the OCR reader with Russian language support
reader = easyocr.Reader(['ru'], gpu=False)

# Mapping dictionaries for character conversion
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

## @brief Checks if the license plate text complies with the required format.
# @param text: The license plate text to check.
# @return: True if the text complies with the format, False otherwise.
def license_complies_format(text):
    if len(text) == 9 and text[0].isalpha() and text[1:4].isdigit() and text[4:6].isalpha() and text[6:].isdigit():
        return True
    elif len(text) == 8 and text[0].isalpha() and text[1:4].isdigit() and text[4:6].isalpha() and text[6:].isdigit():
        return True
    return False

## @brief Formats the license plate text by converting characters using the mapping dictionaries.
# @param text: The license plate text to format.
# @return: The formatted license plate text.
def format_license(text):
    license_plate_ = ''
    for char in text:
        if char in dict_char_to_int:
            license_plate_ += dict_char_to_int[char]
        elif char in dict_int_to_char:
            license_plate_ += dict_int_to_char[char]
        else:
            license_plate_ += char
    return license_plate_

## @brief Reads the license plate text from a cropped image using EasyOCR.
    # @param license_plate_crop: The cropped image of the license plate.
    # @return: A tuple containing the recognized license plate text and its confidence score.
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        # Фильтрация текста по длине и символам
        if len(text) < 6 or len(text) > 9:
            continue

        if license_complies_format(text):
            formatted_text = format_license(text)
            # Дополнительная постобработка для исправления распространенных ошибок
            formatted_text = post_process_license(formatted_text)
            return formatted_text, score
    return None, None

## @brief Post-processes the recognized license plate text to correct common errors.
# @param text: The recognized license plate text.
# @return: The corrected license plate text.
def post_process_license(text):

    corrections = {
        '0': 'О', 'О': '0', 'о': '0', 'о': '0',
        'A': 'А', 'a': 'а', 'B': 'В', 'b': 'в',
        'E': 'Е', 'e': 'е', 'K': 'К', 'k': 'к',
        'M': 'М', 'm': 'м', 'H': 'Н', 'h': 'н',
        'P': 'Р', 'p': 'р', 'C': 'С', 'c': 'с',
        'T': 'Т', 't': 'т', 'Y': 'У', 'y': 'у',
        'X': 'Х', 'x': 'х'
    }
    for char in corrections:
        text = text.replace(char, corrections[char])

    # Дополнительные правила для исправления ошибок
    if len(text) == 9 and text[4] == '0':
        text = text[:4] + 'О' + text[5:]

    return text

    ## @brief Matches a license plate to a vehicle in the tracked vehicles list.
    # @param license_plate: The detected license plate.
    # @param vehicle_track_ids: The list of tracked vehicles.
    # @return: The bounding box and ID of the matched vehicle, or (-1, -1, -1, -1, -1) if no match is found.
def get_car(license_plate, vehicle_track_ids):
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