from click.core import batch
import numpy
from click.testing import CliRunner
from ImgClass.__main__ import main
import pytest
from ImgClass.src import model as m
import tensorflow as tf
from keras.layers import MaxPooling2D
import tensorflow_datasets as tfds
from ImgClass.src import DataHandler as DH
from ImgClass.src import DataClass
from ImgClass.src import model
import os 
import shutil
from ImgClass.src import predict_command
from ImgClass.src import training_command


data = DataClass.Parameters()


def test_predict_function():
    # print("command predict test")
    # training_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/training_set"
    # testing_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/testing_set/CatsDogs"
    # print(os.getcwd())
    
    runner = CliRunner()
    name = "output_testingF"
    training_set = "./testing_data/training_set"
    testing_set = "./testing_data/testing_set"
    data.output_location = "./tests/"+ name

    try:
        os.makedirs(data.output_location)
    except:
        shutil.rmtree(data.output_location)
        os.makedirs(data.output_location)
    
    training_command.run(1, 1, training_set, "", 400, 400, "", -1, data.output_location)
    model = data.output_location + "/Model_Version1"

    #testing with number of epocs
    epocs = 1
    predict_command.run(epocs, 32, "", testing_set, 400, 400, model, -1, data.output_location, True)
    assert os.path.isfile(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location +"/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the number of batches
    batch = 1
    predict_command.run(8, batch, "", testing_set, 400, 400, model, -1, data.output_location, True)
    #print(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.exists(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")


    #predict using confidence_threshold
    ct = 0.5
    predict_command.run(8, 32, training_set, testing_set, 400, 400, model, ct, data.output_location, True)
    assert os.path.exists(data.output_location+ "/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the an output folder
    predict_command.run(8, 32, training_set, testing_set, 400, 400, model, -1, data.output_location, True)
    assert os.path.exists(data.output_location+ "/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the the no report flag
    predict_command.run(8, 32, training_set, testing_set, 400, 400, model, -1, data.output_location, False)

    pass

def test_predict_command():
    # print("command predict test")
    # training_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/training_set"
    # testing_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/testing_set/CatsDogs"
    # print(os.getcwd())
    
    runner = CliRunner()
    name = "output_testingC"
    training_set = "./testing_data/training_set"
    testing_set = "./testing_data/testing_set"
    data.output_location = "./tests/"+ name

    try:
        os.makedirs(data.output_location)
    except:
        shutil.rmtree(data.output_location)
        os.makedirs(data.output_location)
    result = runner.invoke(main, ["train", "-tr", training_set, "-o", data.output_location])
    model = data.output_location + "/Model_Version1"

    #testing with number of epocs
    epocs = 1
    result = runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-e", epocs, "-o", data.output_location])
    assert os.path.isfile(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location +"/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the number of batches
    batch = 1
    result = runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-b", batch, "-o", data.output_location])
    #print(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.exists(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")


    #predict using confidence_threshold
    ct = 0.5
    result = runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-ct", ct, "-o", data.output_location])
    assert os.path.exists(data.output_location+ "/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the an output folder
    result = runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-o", data.output_location])
    assert os.path.exists(data.output_location+ "/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location+"/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the the no report flag
    result = runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-e", epocs, "-o", data.output_location, "--nr", False])

    pass
    
if __name__ == "__main__":
    test_predict_function()
    test_predict_command()


