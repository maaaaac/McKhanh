#!/usr/bin/python

import tensorflow as tf
from tensorflow import keras
import os, sys
import numpy as np
import tempfile
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from tempfile import TemporaryFile


path = "images_processed_small/"
dirs = os.listdir(path)


def process_images():
    for folder in dirs:
        if os.path.isdir(path + folder):
            dirs_folder = os.listdir(path + folder)
            print(path + folder)
            for item in dirs_folder:
                filename = path + folder + "/" + item
                print(filename)
                if os.path.isfile(filename):
                    try:
                        im = Image.open(filename)
                        f, e = os.path.splitext(filename)
                        imResize = im.resize((200, 200), Image.ANTIALIAS)
                        imResizeGrayscale = imResize.convert('L')
                        imResizeGrayscale.save(f + ' resized.jpg', 'JPEG', quality=80)
                    except IOError as e:
                        print(f"An exception occurred {e}")
                    finally:
                        os.remove(filename)


def getDogname(folder_name):
    return folder_name[folder_name.find('-') + 1:]


def generate_data_arrays():
    images_arrays = []
    dognames = []
    label = 1
    for folder in dirs:
        if os.path.isdir(path + folder):
            dogname = getDogname(folder)
            print(dogname)
            dognames.append(dogname)
            dirs_folder = os.listdir(path + folder)
            for item in dirs_folder:
                filename = path + folder + "/" + item
                print(filename)
                if os.path.isfile(filename):
                    try:
                        im = image.imread(filename)
                        arr = im.flatten()
                        arr = np.append(arr, label)
                        images_arrays.append(arr)
                        print(arr)
                    except IOError as e:
                        print(f"An exception occurred {e}")
            label += 1

    return dognames, images_arrays


def data_processing(images_arrays):
    df = np.stack(images_arrays, axis=0)
    np.savetxt('df_small.txt', df, fmt='%d')
    # df = np.loadtxt('df.txt', dtype=np.int64)

    X = df[:, :-1] / 255.0
    y = df[:, -1].reshape(-1, 1)
    enc = OneHotEncoder()
    y = enc.fit_transform(y).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    return X_train, X_test, y_train, y_test


def build_NN(X, y):
    model = Sequential()
    model.add(Dense(4000, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_NN(model, X_train, X_test, y_train, y_test):
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    history = model.fit(X_train, y_train,
                        epochs=100, batch_size=1000, validation_data=(X_test, y_test),
                        shuffle=True, callbacks=cp_callback)


def test(X_train, X_test, y_train, y_test):
    checkpoint_path = "training_1/cp.ckpt"
    model = build_NN(X_train, y_train)
    model.load_weights(checkpoint_path)



def run_NN():
    dognames, images_arrays = generate_data_arrays()
    X_train, X_test, y_train, y_test = data_processing(images_arrays)
    # X_train, X_test, y_train, y_test = data_processing()
    print(f"{X_train.shape, y_train.shape}")
    model = build_NN(X_train, y_train)
    print("model finished")
    train_NN(model, X_train, X_test, y_train, y_test)


def main():
    # test()
    run_NN()


if __name__ == '__main__':
    main()
