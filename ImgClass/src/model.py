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

# I'm not sure if the size of the videos is set, if not these variables can be adaptive. 1 is a place holder
from ImgClass.src import DataClass
from ImgClass.src import DataHandler as DH
import logging
LOGGER = logging.getLogger()



version_num = 1
data = DataClass.Parameters()

# This model takes in an input based on the video size and outputs based on the number of different labels
# in the dataset
def createModel(num_labels):
    LOGGER.info("Generating model")

    x_pixels = data.width_pixels
    y_pixels = data.height_pixels

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape = (x_pixels, y_pixels, 3)))
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
    model.add(tf.keras.layers.Dense(units = num_labels))

    LOGGER.info("Generating model")

    model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])



    data.model = model

    data.model.summary()

    return model

#This function trains the model that is passed in an plots loss and accuracy of the training and validation sets
def trainModel(train_dataset, validation_dataset):
    model = data.model
    num_epochs = data.num_epochs
    LOGGER.info("Training model")
    history = model.fit(
        train_dataset,
        validation_data =validation_dataset,
        epochs = num_epochs
    )

    #I can understand only what's self-explanitory
    LOGGER.info("Finished Training")
    LOGGER.info("Plotting accuracy and loss of training and validation data")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    data.training_accuracy = acc
    data.training_val_accuracy = val_acc

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(num_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    data.plot = plt

# making a prediction based on the model and the images
def makePrediction(image, class_names):
    model = data.model

    image_array = tf.keras.utils.img_to_array(image)

    plotting = image_array

    image_array = tf.expand_dims(image_array, 0)


    predictions = model.predict(image_array)
    print(predictions)
    if len(class_names) == 2:
        score = tf.nn.sigmoid(predictions[0])
    else:
        score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    print(score)


    return class_names[np.argmax(score)], 100 * np.max(score)

