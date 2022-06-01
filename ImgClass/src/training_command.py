import os
from ImgClass.src import DataHandler as DH
from ImgClass.src import DataClass
from ImgClass.src import model as m
from tensorflow import keras
import shutil
import keras.layers
import logging


data = DataClass.Parameters()
try:
    model = keras.models.load_model(data.model_file)
except:
    print()




def run(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV):
    print("I am hereeee")
    if outputV == "Output":
        #had to add this because if the output file didn't exist, it would fail
        # try:
        #     shutil.rmtree(outputV)
        # except:
        #     print("No output file found. Making one.")
        try:
            os.mkdir("Output")
        except:
            print("Output file exists")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=outputV+"/logs.log",
        level=logging.INFO
    )
    LOGGER = logging.getLogger()
    train(epochsV, batchV, trainingV, testingV, heightV, widthV, modelV, ctV, outputV)
    LOGGER.info('Master runs')

    return


def train(numEpocs, numBatchSize, trainingPath, testingPath, height, width, modelPath, conf_thresh_val, output_loc):
    print("training in runfile.")

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

    if modelPath == "":
        print("IF")
        training_d, validation = DH.change_input()
        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(training_d.class_names)
        d.num_confidence = conf_thresh_val
        m.createModel(len(training_d.class_names))
        m.trainModel(training_d, validation)
        model_name = "Model_Version1"
    else:
        model_name = data.model_file.rsplit('/',1)[-1]
        model_name = model_name.rsplit('\\',1)[-1]
        global version_num
        if(model_name.__contains__("Version")):
            version_num = model_name[model_name.index("Version") + 7]
            model_name = model_name[:model_name.index("Version") + 7] + str(int(version_num) + 1) + model_name[(model_name.index("Version") + 8):]
        else:
            model_name = model_name +"Version2"
            version_num = 2
        training_d, validation = DH.change_input()
        if conf_thresh_val == -1:
            conf_thresh_val = 1 / len(training_d.class_names)
        d.num_confidence = conf_thresh_val
        data.model = keras.models.load_model(modelPath)
        data.model.summary()
        m.trainModel(training_d, validation)

    if output_loc == "Output":
        data.model.save(os.getcwd() + "/Output/" + model_name)
        data.plot.savefig(os.getcwd() + "/Output/" + model_name +"/train_data.png")
    else:
        data.model.save(output_loc + "/" + model_name)
        data.plot.savefig(output_loc + "/" + model_name +"/train_data.png")
    return