from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import os
import csv
import cv2
import json
import numpy as np
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Activation, Lambda, Flatten, Dense, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dense(1))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)