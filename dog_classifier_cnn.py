#!/usr/bin/python3

import tensorflow as tf
import os, sys
import numpy as np
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import cv2
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model
from keras_preprocessing.image import ImageDataGenerator
import requests
from bs4 import BeautifulSoup

path = "images_test2/"
dirs = os.listdir(path)

dognames = []

google_image = "https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&"
user_agent = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/102.0.5005.61 Safari/537.36 "
}


def download_images():
    for folder in dirs:
        if os.path.isdir(path + folder):
            dogname = getDogname(folder)
            folder_path = os.path.join(path, folder)
            if dogname.find('_'):
                dogname = dogname.replace('_', ' ')
            dognames.append(dogname)
            download_search(dogname, folder_path)


def download_search(dogname, folder_path):
    url = google_image + 'q=' + dogname

    response = requests.get(url, headers=user_agent)

    html = response.text

    soup = BeautifulSoup(html, 'html.parser')

    results = soup.findAll('img', {'class': ['rg_i Q4LuWd', 'n3VNCb KAlRDb', 'n3VNCb', ]})

    count = 1
    for result in results:
        try:
            print(result['data-src'])
            image_url = result['data-src']
            response = requests.get(image_url)
            image_name = os.path.join(folder_path, dogname + str(count) + '.jpg')
            with open(image_name, 'wb') as fh:
                fh.write(response.content)
            count += 1
            if count > 100:
                break
        except KeyError:
            continue


def getDogname(folder_name):
    return folder_name[folder_name.find('-') + 1:]


def load_data():
    images_arrays = []
    labels = []
    label = 0

    for folder in dirs:
        if os.path.isdir(path + folder):
            # dogname = getDogname(folder)
            # dognames.append(dogname)
            dirs_folder = os.listdir(path + folder)
            print(path + folder)

            for item in dirs_folder:
                filename = os.path.join(path, folder, item)
                print(filename)
                if os.path.isfile(filename):
                    try:
                        im = cv2.imread(filename)
                        if im is not None:
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            imResize = cv2.resize(im, (200, 200))

                            images_arrays.append(imResize)
                            labels.append(label)
                    except IOError as e:
                        print(f"An exception occurred {e}")
            label += 1

    X = np.array(images_arrays, dtype='float32')
    y = np.array(labels, dtype='int32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    return [(X_train, y_train), (X_test, y_test)]


# class CNNModel(Model):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv2D(32, (3, 3), activation='relu')
#         self.pool1 = MaxPooling2D((2, 2))
#         # self.drop1 = Dropout(0.3)
#
#         self.conv2 = Conv2D(64, (3, 3), activation='relu')
#         self.pool2 = MaxPooling2D((2, 2))
#         # self.drop2 = Dropout(0.3)
#
#         self.conv3 = Conv2D(64, (3, 3), activation='relu')
#         self.pool3 = MaxPooling2D((2, 2))
#
#         self.flatten = Flatten()
#         self.d1 = Dense(64, activation='relu')
#         self.pool4 = Dropout(0.3)
#         self.d2 = Dense(10, activation='softmax')
#
#     def call(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         # x = self.drop1(x)
#
#         x = self.conv2(x)
#         x = self.pool2(x)
#         # x = self.drop2(x)
#
#         x = self.conv3(x)
#         x = self.pool3(x)
#
#         x = self.flatten(x)
#         x = self.d1(x)
#         x = self.pool4(x)
#
#         return self.d2(x)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # self.drop2 = Dropout(0.3)

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model


def train_CNN():
    (X_train, y_train), (X_test, y_test) = load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, y_train = shuffle(X_train, y_train, random_state=20)
    model = create_model()

    checkpoint_path = "training_2/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    augment_train = ImageDataGenerator(rotation_range=3,
                                       width_shift_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       brightness_range=[0.1, 0.9])

    history = model.fit(X_train, y_train,
                        epochs=100, batch_size=200, validation_split=0.20, shuffle=True,
                        callbacks=cp_callback)

    history = model.fit(augment_train.flow(X_train, y_train, batch_size=120),
                        epochs=50, validation_data=(X_test, y_test), steps_per_epoch=len(X_train) // 120,
                        callbacks=cp_callback)


def main():
    train_CNN()


if __name__ == '__main__':
    main()
