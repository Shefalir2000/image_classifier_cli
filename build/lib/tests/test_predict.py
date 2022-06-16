from cgi import test
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
    print("command predict test")
    # training_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/training_set"
    # testing_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/testing_set/CatsDogs"
    #print(os.getcwd())

    name = "output_testing_predictF"
    training_set = "./testing_data/training_set"
    testing_set = "./testing_data/testing_set/CatsDogs"
    data.output_location = "./tests/"+ name

    try:
        os.makedirs(data.output_location)
    except:
        shutil.rmtree(data.output_location)
        os.makedirs(data.output_location)
    print("DL", data.output_location)

    runner = CliRunner()
    #runner.invoke(main, ["train", "-tr", training_set, "-e", 1])
    training_command.run(1, 1, training_set, "", 400, 400, "", -1, data.output_location, False)
    #training_command.run(1, 1, training_set, "", 400, 400, "", -1, "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/Here")
    model = data.output_location + "/Model_Version1"

    #testing with number of epocs
    epocs = 1
    predict_command.run(testing_set, model, -1, data.output_location, True, False)
    assert os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the number of batches
    batch = 1
    predict_command.run(testing_set, model, -1, data.output_location , True, False)
    print(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    #assert os.path.exists(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")


    #predict using confidence_threshold
    ct = 0.5
    predict_command.run(testing_set, model, ct, data.output_location , True, False)
    #assert os.path.exists(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the an output folder
    predict_command.run(testing_set, data.output_location + "/Model_Version1", -1, data.output_location , True, False)
    

    #assert os.path.exists(outputLoc+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the the no report flag
    predict_command.run(testing_set, model, -1, data.output_location , False, False)
    assert not os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")


    # predict using the the json flag
    predict_command.run(testing_set, model, -1, data.output_location , False, True)
    assert os.path.isfile(data.output_location + "/Model_Version1/data.json")
    os.remove(data.output_location + "/Model_Version1/data.json")
    pass
    
def test_predict_command():
    print("command predict test")
    # training_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/training_set"
    # testing_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/testing_set/CatsDogs"
    #print(os.getcwd())

    name = "output_testing_predictC"
    training_set = "./testing_data/training_set"
    testing_set = "./testing_data/testing_set/CatsDogs"
    data.output_location = "./tests/"+ name

    try:
        os.makedirs(data.output_location)
    except:
        shutil.rmtree(data.output_location)
        os.makedirs(data.output_location)
    print("DL", data.output_location)

    runner = CliRunner()
    #runner.invoke(main, ["train", "-tr", training_set, "-e", 1])
    training_command.run(1, 1, training_set, "", 400, 400, "", -1, data.output_location, False)
    #training_command.run(1, 1, training_set, "", 400, 400, "", -1, "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/Here")
    model = data.output_location + "/Model_Version1"

    #testing with number of epocs
    epocs = 1
    # predict_command.run(epocs, 32, "", testing_set, 400, 400, model, -1, data.output_location, True)
    runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-o", data.output_location])
    assert os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the number of batches
    batch = 1
    # predict_command.run(8, batch, "", testing_set, 400, 400, model, -1, data.output_location , True)
    runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-o", data.output_location])
    print(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    #assert os.path.exists(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")


    #predict using confidence_threshold
    ct = 0.5
    #predict_command.run(8, 32, training_set, testing_set, 400, 400, model, ct, data.output_location , True)
    runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-ct", ct, "-o", data.output_location])
    #assert os.path.exists(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the an output folder
    #predict_command.run(8, 32, training_set, testing_set, 400, 400, data.output_location + "/Model_Version1", -1, data.output_location , True)
    runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-o", data.output_location])
    #assert os.path.exists(outputLoc+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the the no report flag
    #predict_command.run(8, 32, training_set, testing_set, 400, 400, model, -1, data.output_location , False)
    runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-o", data.output_location, "--nr"])
    assert not os.path.isfile(data.output_location + "/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the the json flag
    runner.invoke(main, ["predict", "-te", testing_set, "-m", model, "-o", data.output_location, "-j", True])
    assert os.path.isfile(data.output_location + "/Model_Version1/data.json")
    # os.remove(data.output_location + "/Model_Version1/data.json")

    pass

if __name__ == "__main__":
    test_predict_function()
    test_predict_command()


