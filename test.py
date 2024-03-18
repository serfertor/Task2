import os

import cv2
import numpy as np

# Загрузка изображения
# image = cv2.imread('Resources/AI - RTUITLab/Photo/00_08.jpg')
image = cv2.imread('Resources/AI - RTUITLab/Random Photos/01_05.jpg')


# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
morph = cv2.dilate(morph, np.ones((5, 5), np.uint8), iterations=2)
morph = cv2.erode(morph, np.ones((5, 5), np.uint8), iterations=1)


contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Original', morph)
cv2.waitKey(0)

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digit_images_path = 'digit_images'
os.makedirs(digit_images_path, exist_ok=True)  # Создание каталога для сохранения вырезанных цифр

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    if aspect_ratio >= 0.35 and area > 650 and area < 50000:
        digit = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.imwrite(f'{digit_images_path}/digit_{x}.jpg', digit)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Определение диапазона зеленого цвета в формате HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Создание маски для определенного диапазона цветов
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Применение маски к исходному изображению
result = cv2.bitwise_and(image, image, mask=mask)

# Отображение и сохранение результата
cv2.imshow('Image with Only Green Color', result)
cv2.imwrite('green_color_only.jpg', result)
# Отображение изображения с выделенными квадратами
cv2.imshow('Digits in Squares', image)
cv2.waitKey(0)
cv2.destroyAllWindows()