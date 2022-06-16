import os
from ImgClass.src import DataHandler as DH
from ImgClass.src import DataClass
from ImgClass.src import model as m
from tensorflow import keras
import shutil
import keras.layers
import logging
import json
from os.path import exists


# initialize all the parameters for this file
data = DataClass.Parameters()

# try to load in the model
try:
    model = keras.models.load_model(data.model_file)
except:
    print()



# the run function for the train command
# this function is useful if you want to add 
# any additional twists to this command
def run(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV, jsonV):
    print("I am hereeee")
    if outputV == "Output":
        try:
            os.mkdir("Output")
        except:
            print("Output file exists")

    #logging base
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=outputV+"/logs.log",
        level=logging.INFO
    )
    data.json = jsonV
    LOGGER = logging.getLogger()
    train(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV)
    LOGGER.info('Master runs')

    return

#function for how to train a model
def train(numEpocs, numBatchSize, trainingPath, testingPath, height, width, modelPath, conf_thresh_val, output_loc):
    #print("training in runfile.")

    # set the data here
    d = DataClass.Parameters()
    d.num_epochs = numEpocs
    d.batch_size = numBatchSize
    d.height_pixels = height
    d.width_pixels = width
    d.training_file = trainingPath
    d.test_file = testingPath
    d.model_file = modelPath
    d.output_location = output_loc

    #if there is no model
    if modelPath == "":
        # convert the data and calculate the conf_thresh if not already provided
        training_d, validation = DH.change_input()
        print(len(training_d) + len(validation))
        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(training_d.class_names)
        d.num_confidence = conf_thresh_val

        #create and train a model
        m.createModel(len(training_d.class_names))
        m.trainModel(training_d, validation)
        model_name = "Model_Version1"
    #if an model already exist
    else:
        # get the model
        model_name = data.model_file.rsplit('/',1)[-1]
        model_name = model_name.rsplit('\\',1)[-1]

        # version control
        global version_num
        if(model_name.__contains__("Version")):
            version_num = model_name[model_name.index("Version") + 7]
            model_name = model_name[:model_name.index("Version") + 7] + str(int(version_num) + 1) + model_name[(model_name.index("Version") + 8):]
        else:
            model_name = model_name +"Version2"
            version_num = 2
        
        #create and train a model also calculate the conf_thresh
        training_d, validation = DH.change_input()
        print(len(training_d) + len(validation))

        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(training_d.class_names)
        d.num_confidence = conf_thresh_val

        #load the model and retrain the model
        data.model = keras.models.load_model(modelPath)
        data.model.summary()
        m.trainModel(training_d, validation)
    # save the output in the correct location
    if output_loc == "Output":
        data.model.save(os.getcwd() + "/Output/" + model_name)
        make_training_json(model_name)
        data.plot.savefig(os.getcwd() + "/Output/" + model_name +"/train_data.png")
    else:
        data.model.save(output_loc + "/" + model_name)
        make_training_json(model_name)
        data.plot.savefig(output_loc + "/" + model_name +"/train_data.png")
    return

def make_training_json(model_name):
    if not data.json:
        return
    if exists(data.output_location + "/" + model_name + "/data.json"):
        with open (data.output_location + "/" + model_name + "/data.json", 'r+') as jason:
            arr = json.loads(jason.read())
    else:
        arr = {}

    with open(data.output_location + "/" + model_name + "/data.json", 'w') as jason:
        num_file = count_files(data.training_file, 0) 
        val_files= int(num_file * .2)
        train_files = num_file - val_files
        arr['Train Parameters'] = []
        arr['Train Parameters'].append({"Epoch Number": data.num_epochs,
        "Batch Size": data.batch_size,
         "Height Pixels": data.height_pixels,
         "Width Pixels" : data.width_pixels,
         "Training File": data.training_file,
         "Number of Training Files": train_files,
         "Number of Validation Files": val_files
          })


  
        if data.model_file != "":
            arr["Train Parameters"][0]["Inital Model"] = data.model_file
        
        arr['Train Results'] = []
        arr['Train Results'].append({})

        dir_path = data.training_file
        count = 0


        for i in range(data.num_epochs):
            arr["Train Results"][0]["epoch " + str(i + 1)] = "validation accuracy " + str(data.training_val_accuracy[i])  + " training accuracy " + str(data.training_accuracy[i])

        json.dump(arr, jason, indent=2)

def count_files(data_path, count):
    for path in os.listdir(data_path):
        if os.path.isdir( data_path + '/' +path):
            count = count_files(data_path + '/' + path, count)
        elif not "jpeg" in path and not "jpg" in path and not "jfif" in path and not "pjpeg" in path and not "pjp" in path and not "png" in path and not "svg" in path and not "webp" in path:
            return count 
        elif os.path.isfile(data_path + '/' + path):
            count = count + 1
    return count 



