import tensorflow as tf
import PIL
import pathlib
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv
from keras.layers import MaxPooling2D
from tensorflow import keras
from keras import models
import keras.layers
# I'm not sure if the size of the videos is set, if not these variables can be adaptive. 1 is a place holder
from hello.src import DataClass
from hello.src import DataHandler as DH

"""
TODO:
Make sure all different size images in the input parameter sizes change

other solution: insert parameters in the application
"""




data = DataClass.Parameters()
try:
    model = keras.models.load_model(data.model_file)
except:
    print()
# This model takes in an input based on the video size and outputs based on the number of different labels
# in the dataset
def createModel(num_labels):
    x_pixels = data.width_pixels
    y_pixels = data.height_pixels

    global model

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape = (x_pixels, y_pixels, 3)))
    model.add(tf.keras.layers.Rescaling(1./255))
    model.add(tf.keras.layers.Conv2D(16, 3, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding = 'same' , activation='relu'))
    model.add(MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units = num_labels))



    model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])

    model.summary()

    return model

#This function trains the model that is passed in an plots loss and accuracy of the training and validation sets
def trainModel(train_dataset, validation_dataset):
    num_epochs = data.num_epochs
    history = model.fit(
        train_dataset,
        validation_data =validation_dataset,
        epochs = num_epochs
    )
    print(train_dataset.class_names)
    #I can understand only what's self-explanitory
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

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
    plt.show()

#def makePrediction(model, image, class_names):
def makePrediction( image, class_names):
    """height = 400
    width = 400
    img = tf.keras.utils.load_img(image, target_size = (height, width))

    image_array = tf.keras.utils.img_to_array(img)"""



    #Hard code passng in an image for testing purposes
    # img = tf.keras.utils.load_img(
    #     "test1.jpg", target_size = (400,400)
    # )
    # image_array = tf.keras.utils.img_to_array(img)

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

def runNewModel(numEpocs, numBatchSize, trainingPath, testingPath, height, width, modelPath):
    # set the data here
    d = DataClass.Parameters()
    #numEpocs, numBatchSize, height, width, trainingPath, testingPath, modelPath
    d.num_epochs = numEpocs
    d.batch_size = numBatchSize
    d.height_pixels = height
    d.width_pixels = width
    d.training_file = trainingPath
    d.test_file = testingPath
    d.model_file = modelPath
    if modelPath == "":
        training_d, validation = DH.change_input()
        model = createModel(len(training_d.class_names))
        trainModel(training_d, validation)
        DH.categorize(0.6, training_d)
    else:
        training_d, validation = DH.change_input()
        model = keras.models.load_model(modelPath)
        model.summary()
        trainModel(training_d, validation)
        DH.categorize(0.6, training_d)
    return



