# -*- coding: utf-8 -*-
"""dog_cnn

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VcGcJIai3GJVx2T1vDg_rgKF2URyhGIZ
"""

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
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import Model
from keras_preprocessing.image import ImageDataGenerator
import requests
from bs4 import BeautifulSoup

from google.colab import drive
drive.mount('/content/gdrive')

# !unzip gdrive/My\ Drive/data/low-resolution.zip

# !unzip gdrive/My\ Drive/data/test.zip

path = "test2/"
dirs = os.listdir(path)

dognames = []

def getDogname(folder_name):
    return folder_name[folder_name.rfind('-') + 1:]


def load_data():
    images_arrays = []
    labels = []
    label = 0
    folder_count = 0

    for folder in dirs:
        if os.path.isdir(path + folder):
            dogname = getDogname(folder)
            dognames.append(dogname)
            dirs_folder = os.listdir(path + folder)
            print(path + folder)
            dog_counter = 1
            for item in dirs_folder:
                filename = os.path.join(path, folder, item)
                print(filename)
                if os.path.isfile(filename):
                    try:
                      im = cv2.imread(filename)
                      if im is not None:
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            imResize = cv2.resize(im, (224, 224))

                            images_arrays.append(imResize)
                            labels.append(label)
                            dog_counter += 1
                            if dog_counter == 3000:
                                break
                    except IOError as e:
                        print(f"An exception occurred {e}")
            folder_count += 1
            if folder_count == 5:
                break
            label += 1

    X = np.array(images_arrays, dtype='float32')
    y = np.array(labels, dtype='int32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return [(X_train, y_train), (X_test, y_test)]

(X_train, y_train), (X_test, y_test) = load_data()

def create_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # model.add(Conv2D(16, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='same',))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.2))

    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
    # model.add(MaxPooling2D((2, 2), padding='same',))
    # model.add(Dropout(0.3))

    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.3))

    # model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
    # model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.3))

    # model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
    # model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.3))

    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(5, activation='softmax'))

    model.summary()
    return model

dognames

X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, y_train = shuffle(X_train, y_train, random_state=20)
model = create_model()

checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)
model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

augment_train = ImageDataGenerator(rotation_range=3,
                                width_shift_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                brightness_range=[0.1, 0.9])

estop = tf.keras.callbacks.EarlyStopping( monitor="val_loss",  patience=10,
                                  verbose=1, 
                                  restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                      patience=4, min_lr=0.000001, verbose=1)

# history = model.fit(X_train, y_train,
#                     epochs=100, batch_size=64, validation_split=0.15, 
#                     shuffle=True,
#                     callbacks=[checkpoint, estop, reduce_lr])

history = model.fit(X_train, y_train,
                epochs=100, batch_size=64, validation_data=(X_test, y_test), 
                shuffle=True,
                callbacks=[checkpoint, estop, reduce_lr])

# history = model.fit(augment_train.flow(X_train, y_train, batch_size=64),
#                     epochs=50, validation_data=(X_test, y_test), 
#                     steps_per_epoch=len(X_train) // 64,
#                     callbacks=[checkpoint, estop, reduce_lr])

model = create_model()

from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model.load_weights('training_1/cp.ckpt')

X_test = np.load("X_test.txt.npy")
y_test = np.load("y_test.txt.npy")

model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

confusion_matrix(y_test, np.argmax(y_pred, axis=1))

dognames = ['chinese_rural_dog', 'golden_retriever', 'Bichon_Frise', 'Cardigan', 'teddy']

fig, ax = plt.subplots(figsize=(15, 10))
cmp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, np.argmax(y_pred, axis=1)),
    display_labels=dognames,
)

cmp.plot(ax=ax)
plt.show();

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(history.history["accuracy"])
ax1.plot(history.history['val_accuracy'])
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax1.set(xlabel="Epoch", ylabel="Accuracy")
ax2.set(xlabel = "Epoch", ylabel="Loss")
fig.suptitle("model accuracy")
# fig.ylabel("Accuracy")
# fig.xlabel("Epoch")
ax1.legend(["Accuracy","Validation Accuracy"])
ax2.legend(["Loss","Validation Loss"])
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, np.argmax(y_pred, axis=1), target_names=dognames))