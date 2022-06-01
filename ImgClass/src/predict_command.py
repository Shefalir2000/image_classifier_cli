import logging
import os
from ImgClass.src import DataHandler as DH
from ImgClass.src import DataClass
from ImgClass.src import model as m
from tensorflow import keras
import keras.layers
import shutil

#initialize all the parameters for this file
data = DataClass.Parameters()
#try to load in the model
try:
    model = keras.models.load_model(data.model_file)
except:
    print()

# the run function for the predict command
# this function is useful if you want to add 
# any additional twists to this command
def run(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV, make_reportV):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename= outputV + "/logs.log",
        level=logging.INFO
    )
    LOGGER = logging.getLogger()
    predict(epochsV, batchV, testingV, heightV, widthV, modelV, ctV, outputV, make_reportV)
    LOGGER.info('Master runs')

    return

# function that organizes all the predict parameters and 
# makes the initial call for predict
def predict(numEpocs, numBatchSize, testingPath, height, width, modelPath, conf_thresh_val, output_loc, make_reportV):
    """print("testing in runfile.")
    if output_loc == "Output":
        print("Bleh")"""
    
    # set the data here
    d = DataClass.Parameters()
    d.num_epochs = numEpocs
    d.batch_size = numBatchSize
    d.height_pixels = height
    d.width_pixels = width
    d.test_file = testingPath
    d.model_file = modelPath
    d.output_location = output_loc
    d.model = keras.models.load_model(d.model_file)
    d.make_report = make_reportV

    # throws exception if the modelPath is empty
    if modelPath == "":
        raise Exception("You must have a model to predict values.")

    # makes a list of the classes that are possible labels 
    else:
        numClasses = list()
        #checks makes the list of all the labels
        for i in os.listdir(testingPath):

            #Fixed bug here where num classes would be empty due to i alone not being a valid directory
            if os.path.isdir(testingPath + "/" + i):
                numClasses.append(i)
            
        # calculates the confidence threshold based on the number of labels
        # only calculates if an estimated num_conf is not given
        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(numClasses)
        d.num_confidence = conf_thresh_val
        
        # loads the model and categorizes the images
        m.model = keras.models.load_model(modelPath)
        DH.categorize(d.num_confidence, numClasses)

    return
