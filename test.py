import shutil

import tensorflow as tf
import PIL
import pathlib
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv
import os
from keras.layers import MaxPooling2D
from tensorflow import keras
from keras import models
import keras.layers
from keras.layers import Rescaling

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape = (400, 400, 3)))
model.add(tf.keras.layers.Rescaling(1./255))
model.add(tf.keras.layers.Conv2D(16, 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, 3, padding = 'same' , activation='relu'))
model.add(MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, 3, padding = 'same' , activation='relu'))
model.add(MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, 3, padding = 'same' , activation='relu'))
model.add(MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units = 112321))


model.compile(optimizer = 'adam',
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy'])

model.summary()

print(model.layers[-1].output.get_shape()[1])