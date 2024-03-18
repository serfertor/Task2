import os
import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('Resources/AI - RTUITLab/Photo/00_08.jpg')
# image = cv2.imread('Resources/AI - RTUITLab/Random Photos/01_05.jpg')

image_itog = image.copy()

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 241, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
morph = cv2.dilate(morph, np.ones((5, 5), np.uint8), iterations=2)
morph = cv2.erode(morph, np.ones((5, 5), np.uint8), iterations=1)

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

new_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if 500 < area < 50000:
        new_contours.append((x, y, w, h))

corners = []
for i in range(len(new_contours)):
    for j in range(i + 1, len(new_contours)):
        x11 = new_contours[i][0]
        x21 = new_contours[j][0]
        if max (x11, x21) - min(x11, x21) <= 15:
            y11 = new_contours[i][1]
            y21 = new_contours[j][1]
            x12 = x11 + new_contours[i][2]
            x22 = x21 + new_contours[j][2]
            y12 = y11 + new_contours[i][3]
            y22 = y21 + new_contours[j][3]
            corners.append([(min(x11, x21),
                             min(y11, y21)),
                            (max(x12, x22),
                             max(y12, y22))])
for i in corners:
    cv2.rectangle(image, i[0], i[1], (0, 255, 0), 2)


contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# digit_images_path = 'digit_images'
# os.makedirs(digit_images_path, exist_ok=True)  # Создание каталога для сохранения вырезанных цифр

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    if aspect_ratio >= 0.35 and 650 < area < 50000:
        # digit = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imwrite(f'{digit_images_path}/digit_{x}.jpg', digit)


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()