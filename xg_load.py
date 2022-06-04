
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16

SIZE = 256

LABELS = ['Bichon Frise', 'Samoyed', 'Husky', 'Golden Retriever', 'Teddy']

# process the input image
img_path = glob.glob("input_images/*")[0]
label = img_path.split("/")[-1]
img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
img = cv2.resize(img, (SIZE, SIZE))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = img / 255.0

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model('xg_dog.json')

input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction = model.predict(input_img_features)[0] 
print("The prediction for this image is: \n\n\n    ", LABELS[prediction])
print('\n')
