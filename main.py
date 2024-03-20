import cv2
import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import PIL
from PIL import Image
from matplotlib import pyplot as plt

def get_digits(filepath):
    # Загрузка изображения
    image = cv2.imread(filepath)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 241, 255, cv2.THRESH_BINARY)

    # Применение морфологических изменений
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.dilate(morph, np.ones((5, 5), np.uint8), iterations=2)
    morph = cv2.erode(morph, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []
    digits = []
    # Поиск подходящих контуров и их координаты
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.35 and 650 < area < 30000:
            new_contours.append((x, y, w, h, area))

    corners = []
    # Расчет близлежащих контуров
    for i in range(len(new_contours)):
        for j in range(i + 1, len(new_contours)):
            x11 = new_contours[i][0]
            x21 = new_contours[j][0]
            if max(x11, x21) - min(x11, x21) <= 12:
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

    j = 0
    if len(corners) != 4:
        new_contours.sort(key=lambda x: x[4], reverse=True)
        for i in range(len(corners), 4):
            corners.append([(new_contours[j][0], new_contours[j][1]),
                            (new_contours[j][0] + new_contours[j][2], new_contours[j][1] + new_contours[j][3])])
            j += 1

    # Обрезка отдельных цифр
    for i in corners:
        digits.append(cv2.bitwise_not(morph[i[0][1]:i[1][1], i[0][0]:i[1][0]]))
    j = 0
    for i in digits:
        cv2.imwrite(f"{j}.jpg", i)
        j += 1
    return digits


def train_model():
    # Гиперпараметры модели
    num_classes = 10  # число классов - число цифр
    input_shape = (28, 28, 1)  # размер изображений цифр, они не цветные, поэтому канал 1.

    # загружаем данные (изображения и их классы), отдельно обучающие и тестовые
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  #

    # Преобразуем во float и диапазон [0, 1]
    x_train = x_train.astype("float32") / 255  #
    x_test = x_test.astype("float32") / 255  #

    x_train = np.expand_dims(x_train, -1)  # для обучающих
    x_test = np.expand_dims(x_test, -1)  # для тестовых

    # переводим метки классов в унитарные вектора
    y_train = keras.utils.to_categorical(y_train, num_classes)  # обучающие
    y_test = keras.utils.to_categorical(y_test, num_classes)  # тестовые

    model = keras.Sequential(  # слои перечисляются ниже
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),  #
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),  #
            layers.MaxPooling2D(pool_size=(2, 2)),  #
            layers.Flatten(),  #
            layers.Dense(num_classes, activation="softmax"),  #
        ]
    )
    batch_size = 128  # размер пакета (batch)
    epochs = 5  # количество эпох обучения

    # задаем функцию ошибки, метод обучения и метрику проверки
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])  # функция ошибки на тестовых данных
    print("Test accuracy:", score[1])  # метрика (из заданных, у нас accuracy) на тестовых данных

    test_example = 12  # индекс примера
    test_input = x_test[test_example:test_example + 1]  # изображение этого примера
    test_output = np.argmax(model.predict(test_input), axis=-1)

    print('Это цифра', test_output)  # выводим номер класса, он же название цифры

    model.save("model.keras")


def recognize_time(pathfile, model):
    num = get_digits(pathfile)
    digit = num[3]
    digit = cv2.resize(digit, (64, 64), interpolation=cv2.INTER_LINEAR)
    digit = tf.keras.utils.img_to_array(digit)
    digit /= 255.0
    digit = tf.expand_dims(digit, 0)  # Create a batch

    print(digit.shape)
    print(digit)
    print(np.argmax(tf.nn.softmax(model.predict(digit)[0])))


if __name__ == '__main__':
    # image = cv2.imread('Resources/AI - RTUITLab/Photo/00_08.jpg')
    # image = cv2.imread('Resources/AI - RTUITLab/Random Photos/01_05.jpg')
    model = keras.models.load_model("model.keras")
    get_digits("Resources/AI - RTUITLab/Photo/01_42.jpg")


    img = tf.keras.utils.load_img(
        "2.jpg", target_size=(28, 28), color_mode="grayscale"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"][np.argmax(score)], 100 * np.max(score))
    )
    # while True:
    #     print("Введите 1 - для ввода пути до файла вручную \n"
    #           "Введите 2 - для ввода пути до директории \n"
    #           "Введите 0 - для выхода из программы")
    #     text = input()
    #     match text:
    #         case '0':
    #             break
    #         case '1':
    #             print ("Введите путь до файла")
    #             path = input()
    #             recognize_time(path, model)
    #         case _:
    #             continue
