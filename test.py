import os

import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('Resources/AI - RTUITLab/Random Photos/01_05.jpg')


# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

morph = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

canny = cv2.Canny(morph, 50, 175, apertureSize=3)

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cv2.imshow('Original', canny)
cv2.waitKey(0)


digit_images_path = 'digit_images'
os.makedirs(digit_images_path, exist_ok=True)  # Создание каталога для сохранения вырезанных цифр

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    if aspect_ratio > 0.2 and aspect_ratio < 2.5 and area > 200:
        digit = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(f'{digit_images_path}/digit_{x}.jpg', digit)

# Отображение изображения с выделенными квадратами
cv2.imshow('Digits in Squares', image)
cv2.waitKey(0)
cv2.destroyAllWindows()