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
    training_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/training_set"
    testing_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/testing_set/CatsDogs"
    print(os.getcwd())
    runner = CliRunner()
    #runner.invoke(main, ["train", "-tr", training_set, "-e", 1])
    training_command.run(1, 1, training_set, "", 400, 400, "", -1, "Output")
    training_command.run(1, 1, training_set, "", 400, 400, "", -1, "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/Here")
    model = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/Output/Model_Version1"

    #testing with number of epocs
    epocs = 1
    predict_command.run(epocs, 32, "", testing_set, 400, 400, model, -1, "Output", True)
    assert os.path.isfile(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the number of batches
    batch = 1
    predict_command.run(8, batch, "", testing_set, 400, 400, model, -1, "Output", True)
    print(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    #assert os.path.exists(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")


    #predict using confidence_threshold
    ct = 0.5
    predict_command.run(8, 32, training_set, testing_set, 400, 400, model, ct, "Output", True)
    #assert os.path.exists(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(os.getcwd()+"/Output/Model_Version1/Confidence and Accuracy Report.md")

    # predict using the an output folder
    outputLoc = os.getcwd() + "/tests/Here"
    predict_command.run(8, 32, training_set, testing_set, 400, 400, "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/Here/Model_Version1", -1, outputLoc, True)
    

    #assert os.path.exists(outputLoc+"/Output/Model_Version1/Confidence and Accuracy Report.md")
    assert os.path.isfile(outputLoc+"/Model_Version1/Confidence and Accuracy Report.md")
    os.remove(outputLoc+"/Model_Version1/Confidence and Accuracy Report.md")
    # predict using the the no report flag
    predict_command.run(8, 32, training_set, testing_set, 400, 400, model, -1, "Output", False)
    
if __name__ == "__main__":
    test_predict_function()


