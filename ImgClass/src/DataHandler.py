import shutil

import tensorflow as tf
import os
from tensorflow import keras
from matplotlib import pyplot as plt
import random as r

from ImgClass.src import DataClass
from ImgClass.src import model as m

import logging
LOGGER = logging.getLogger()

import mdutils
from mdutils import Html
from mdutils.mdutils import MdUtils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#dataclass can be changed in anoteher class
data  = DataClass.Parameters()

def change_input():
    #check the different folders in the original input file name
    LOGGER.info("Converting directory into training and validation datasets")
    input_file = data.training_file
    #print(input_file)
    batch_size = data.batch_size
    image_size = (data.height_pixels, data.width_pixels)
    seedNum = r.randint(1,10000)
    
    #convert the data into TFRecord Files so they can be easily processed
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_file,
        validation_split=0.2,
        subset="training",
        seed=seedNum,
        image_size=image_size,
        batch_size=batch_size,
    )
    #print(str(len(list(train_ds))) + "LENTGHT")
    print(str(sum(1 for _ in train_ds)))
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_file,
        validation_split=0.2,
        subset="validation",
        seed=seedNum,
        image_size=image_size,
        batch_size=batch_size,
    )

    return train_ds, val_ds

# resize the images
def scale_resize(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image,(224,224))
    return (image,label)

# resize the data set
def scale_resize_dataset(dataset):
    ds = (
        dataset
        .map(scale_resize, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        .batch(data.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return ds

# testing the predictions to check accuracy
def testing(file_name, path, height, width,modelInput,train_ds):
    
    path_file = path + "/" + file_name
    img = tf.keras.utils.load_img(path_file, target_size = (data.height_pixels, data.width_pixels))
    img_array = tf.keras.utils.img_to_array(img)
    predictions = m.makePrediction(modelInput, img_array, train_ds.class_names)
    return predictions

#gathering the data confidence
def gathering_data_confidence(train_ds):
    height = data.height_pixels
    width = data.width_pixels
    model = data.model_file
    test_directory = data.test_file
    count = 0
    predict_colorsV2 = list()
    predict_colorsV3 = list()
    LOGGER.info("Making predictions on V2 and V3 colors")
    for directory_name in test_directory: #colorsV# directory
        count += 1
        for testing_name in train_ds.class_names: #red, green, blue
            path = test_directory + "/" + testing_name
            files = os.listdir(path)
            for file_name in files: #files in red,green, or blue directory
                
                prediction = m.makePrediction(path, train_ds.class_names)
                if count == 1:
                    # add the prediction for that file in the colorV2 list
                    predict_colorsV2.append(prediction)
                else:
                    # add the prediction for that file in the colorV3 list
                    predict_colorsV3.append(prediction)

#categorizing the images and making the markdown file
def categorize(confidence_threshold, class_names):
    testing_directory_name = data.test_file

    LOGGER.info("Making predictions on test dataset and organizing entries into confidence directories")

    above_threshold = list()
    below_threshold = list()
    print("list",testing_directory_name)
    for sub in os.listdir(testing_directory_name):
        print("sub",sub)
        for file in os.listdir(testing_directory_name + "/" + sub):
            print(file)
            img = tf.keras.utils.load_img(testing_directory_name + "/" + sub + "/" + file, target_size=(data.width_pixels, data.height_pixels))
            img_array = tf.keras.utils.img_to_array(img)
            prediction, confidence = m.makePrediction( img_array, class_names)
            if confidence < confidence_threshold*100:
                # add the values to the arrays
                below_threshold.append((confidence, prediction, sub, testing_directory_name + "/" + sub + "/" +file))
            else:
                above_threshold.append((confidence, prediction, sub, testing_directory_name + "/" + sub + "/" + file))
    #print("AT", above_threshold)
    above_avg_accuracy, above_avg_confidence = caluclate_average(above_threshold)
    below_avg_accuracy, below_avg_confidence = caluclate_average(below_threshold)
    data.accuracy_below = below_avg_accuracy
    data.accuracy_above = above_avg_accuracy
    data.test_files_num = len(below_threshold) + len(above_threshold)
    if(data.make_report):
        mdFile = MdUtils(file_name=data.model_file + "/Confidence and Accuracy Report",
                         title="Confidence and Accuracy Report")

        mdFile.new_header(level = 1 ,title = "Model version number " + str(m.version_num))

        mdFile.new_header(level=1, title="Threshold Value")
        mdFile.new_paragraph("The threshold value is "+ str(confidence_threshold*100)+ "." )
        mdFile.new_header(level=2, title="Confidence/Accuracy Above the Threshold")
        mdFile.new_paragraph("The average confidence level above the threshold is: " + str(above_avg_confidence))
        mdFile.new_paragraph("The average accuracy level above the threshold is: " + str(above_avg_accuracy))
        mdFile.new_paragraph("The data is in Appendix A")

        mdFile.new_header(level=3, title="Confidence/Accuracy Below the Threshold")
        mdFile.new_paragraph("The average confidence level below the threshold is: " + str(below_avg_confidence))
        mdFile.new_paragraph("The average accuracy level below the threshold is: " + str(below_avg_accuracy))
        mdFile.new_paragraph("The data is in Appendix B")

        #Check if training file exist in model
        if os.path.exists(data.model_file + "/train_data.png"):
            mdFile.new_header(level = 2, title = "Training Accuracy and Loss for Validation vs Training data")
            mdFile.new_line(mdFile.new_inline_image(text="train_data.png", path = "/train_data.png"))
        else:
            print("data image not found")
        mdFile.new_header(level=2, title="Appendix A")
        for i in above_threshold:
            mdFile.write("Path to Image: "+ str(i[3])+"\t"+"Confidence Level: " + str(i[0])+"\t"+"Predicted Label: "+str(i[1])+"\t"+"Actual Label: "+str(i[2])+"\n")

        mdFile.new_header(level=2, title="Appendix B")
        for i in below_threshold:
            mdFile.write("Path to Image: "+ str(i[3])+"\t"+"Confidence Level: " + str(i[0])+"\t"+"Predicted Label: "+str(i[1])+"\t"+"Actual Label: "+str(i[2])+"\n")

        mdFile.create_md_file()
        LOGGER.info("Report generated")
    LOGGER.info("Finished predicting test data ")

#caluclate the average accuracy and confidence for the markdown file
def caluclate_average(threshold_list):
    avg_confidence = 0
    avg_accuracy = 0
    counter = 0
    for i in threshold_list:
        counter += 1
        avg_confidence += i[0]
        if i[1] == i[2]:
            avg_accuracy += 1
    if counter == 0:
        return 0, 0
    return avg_accuracy/counter, avg_confidence/counter



