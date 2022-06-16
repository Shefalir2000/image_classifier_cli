import logging
from operator import ge
import os
from unicodedata import name

from black import InvalidInput
from ImgClass.src import DataHandler as DH
from ImgClass.src import DataClass
from ImgClass.src import model as m
from tensorflow import keras
import keras.layers
import shutil
import json
from os.path import exists

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
# def run(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV, make_reportV):
def run(testingV, modelV, ctV, outputV, make_reportV, jsonV, UnlabeledV):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename= outputV + "/logs.log",
        level=logging.INFO
    )
    data.json = jsonV
    LOGGER = logging.getLogger()
    if not UnlabeledV:
        predict(testingV, modelV, ctV, outputV, make_reportV)
    else:
        predict_no_labels(testingV, modelV, ctV, make_reportV)
    LOGGER.info('Master runs')

    return

# function that organizes all the predict parameters and 
# makes the initial call for predict
def predict( testingPath, modelPath, conf_thresh_val, outputV, make_reportV):
    """print("testing in runfile.")
    if output_loc == "Output":
        print("Bleh")"""
    
    # set the data here
    d = DataClass.Parameters()
    d.test_file = testingPath
    d.model_file = modelPath
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
            conf_thresh_val = 100 / len(numClasses)
        d.num_confidence = conf_thresh_val
        
        # loads the model and categorizes the images
        m.model = keras.models.load_model(modelPath)
        DH.categorize(d.num_confidence/100, numClasses)
        make_predict_json()

    return

def make_predict_json():
    if not data.json:
        return
    if exists(data.model_file + "/data.json"):
        with open(data.model_file + "/data.json", 'r+') as jason:
            datas = json.loads(jason.read())

    else:
        datas = {}

    datas['Predict Parameters'] = []
    datas['Predict Parameters'].append({
        "Confidence Threshold": data.num_confidence,
        "Model File": data.model_file,
        "Log Output Location": data.output_location
    })
    datas['Predict Results'] = []
    if not data.unlabeled:
        datas['Predict Results'].append({
            "Number of Testing Files": data.test_files_num,
            "Below Threshold Accuracy": data.accuracy_below,
            "Above Threshold Accuracy": data.accuracy_above
        })
    else:
        datas["Predict Results"].append({
            "Number of Testing Files": data.test_files_num,          
        })


    with open(data.model_file + "/data.json", 'w') as jason:
        json.dump(datas,jason, indent=2)

def predict_no_labels(testingPath, modelPath, conf_thresh_val, make_reportV):
    """print("testing in runfile.")
    if output_loc == "Output":
        print("Bleh")"""
    
    # set the data here
    d = DataClass.Parameters()
    d.test_file = testingPath
    d.model_file = modelPath
    d.model = keras.models.load_model(d.model_file)
    d.make_report = make_reportV

    # throws exception if the modelPath is empty
    if modelPath == "":
        raise Exception("You must have a model to predict values.")

    # makes a list of the classes that are possible labels 
    else:
        nameClasses = text_parse()
        if not d.model.layers[-1].output.get_shape()[1] == len(nameClasses):
            raise InvalidInput("Must have same number of classes as initially trained on")
        if(conf_thresh_val == -1):
            conf_thresh_val = 100 / len(nameClasses)
        d.num_confidence = conf_thresh_val
        DH.categorizeUL(d.num_confidence/100, len(nameClasses))
        # numClasses = list()
        # #checks makes the list of all the labels
        # for i in os.listdir(testingPath):

        #     #Fixed bug here where num classes would be empty due to i alone not being a valid directory
        #     if os.path.isdir(testingPath + "/" + i):
        #         numClasses.append(i)
            
        # # calculates the confidence threshold based on the number of labels
        # # only calculates if an estimated num_conf is not given
        # if conf_thresh_val == -1:
        #     conf_thresh_val = 50
        # d.num_confidence = conf_thresh_val
        
        # # loads the model and categorizes the images
        # m.model = keras.models.load_model(modelPath)
        # DH.categorize(d.num_confidence/100, numClasses)
        # make_predict_json()
    pass


def text_parse():
    nameList = list()
    File_object = open(data.name_classes, "r")
    for line in File_object:
            name = line.strip().split(',')
            print(name)
            nameList+= name
    File_object.close()
    return nameList
