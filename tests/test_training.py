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
from ImgClass.src import training_command
import shutil
import os

data = DataClass.Parameters()

def test_train_function():
    print("function train test")
    # training_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/training_set"
    # print("command train test")

    name = "output_testingF"
    training_set = "./testing_data/training_set"
    data.output_location = "./tests/"+ name

    try:
        os.makedirs(data.output_location)
    except:
        shutil.rmtree(data.output_location)
        os.makedirs(data.output_location)
    print("DL", data.output_location)
    
    #testing with number of epocs
    epochs = 1
    result = training_command.run(epochs, 32, training_set, "", 400, 400, "", -1, data.output_location, False)
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        print(i)
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
            
    shutil.rmtree(data.output_location+"/Model_Version1")
    #testing with batch size changed
    batch = 1
    result =training_command.run(8, batch, training_set, "", 400, 400, "", -1, data.output_location, False)
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    
    shutil.rmtree(data.output_location+"/Model_Version1")
    #testing with height and width
    height = 100
    width = 100

    result =training_command.run(8, 32, training_set, "", height, width, "", -1, data.output_location, False)
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    shutil.rmtree(data.output_location+"/Model_Version1")
    #testing with a confidence threshold
    ct = 0.5

    result = training_command.run(8, 32, training_set, "", 400, 400, "", ct, data.output_location, False)
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    #testing with an output location 
    result = training_command.run(8, 32, training_set, "", 400, 400, "", -1, data.output_location, False)
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False

    #testing with model
    model = data.output_location+"/Model_Version1"
    result = training_command.run(8, 32, training_set, "", 400, 400, model, -1, data.output_location, False)
    assert os.path.exists(data.output_location+"/Model_Version2")
    assert os.path.isdir(data.output_location+"/Model_Version2")
    for i in os.listdir(data.output_location+"/Model_Version2"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    shutil.rmtree(data.output_location+"/Model_Version1")
    shutil.rmtree(data.output_location+"/Model_Version2")
    # shutil.rmtree(outputLoc+"/Model_Version1")
    pass
    

def test_train_command():
    #training_set = "C:/Users/sranjan31/Source/clitoolpleasework/image_classifier_cli/tests/training_set"
    # print("command train test")
    # print(os.getcwd())
    runner = CliRunner()
    name = "output_testingC"
    training_set = "./testing_data/training_set"
    data.output_location = "./tests/"+ name
    #print(os.getcwd())
    try:
        os.makedirs(data.output_location)
    except:
        shutil.rmtree(data.output_location)
        os.makedirs(data.output_location)
    #print("LOOK HERE PLEASE", data.output_location)
    #testing with number of epocs
    epocs = 1
    #outputLoc = os.getcwd() + "/tests/Here"
    print(data.output_location)
    result = runner.invoke(main, ["train", "-tr", training_set, "-e", epocs, "-o", data.output_location])
    #print("HIIIIIIII", result.output)
    #print("LOOK HERE PLEASE 2", data.output_location)
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        #print(i)
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    assert data.model != None   
    shutil.rmtree(data.output_location+"/Model_Version1")
    
    # testing with batch size changed
    batch = 1
    result = runner.invoke(main, ["train", "-tr", training_set, "-b", batch, "-o", data.output_location])
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    assert data.model != None 
    shutil.rmtree(data.output_location+"/Model_Version1")
    #testing with height and width
    height = 100
    width = 100

    result = runner.invoke(main, ["train", "-tr", training_set, "-h", height, "-w", width, "-o", data.output_location])
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    assert data.model != None 
    shutil.rmtree(data.output_location+"/Model_Version1")
    #testing with a confidence threshold
    ct = 0.5

    result = runner.invoke(main, ["train", "-tr", training_set, "-ct", ct, "-o", data.output_location])
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    assert data.model != None 
    shutil.rmtree(data.output_location+"/Model_Version1")
    #testing with an output location 
    #outputLoc = os.getcwd() + "/tests/Here"
    result = runner.invoke(main, ["train", "-tr", training_set, "-o", data.output_location])
    #print(outputLoc+"/Model_Version1")
    assert os.path.exists(data.output_location+"/Model_Version1")
    assert os.path.isdir(data.output_location+"/Model_Version1")
    for i in os.listdir(data.output_location+"/Model_Version1"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    assert data.model != None 
    #testing with model
    model = data.output_location+"/Model_Version1"
    result = runner.invoke(main, ["train", "-tr", training_set, "-m", model, "-o", data.output_location])
    assert os.path.exists(data.output_location+"/Model_Version2")
    assert os.path.isdir(data.output_location+"/Model_Version2")
    for i in os.listdir(data.output_location+"/Model_Version2"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png":
            assert True
        else:
            assert False
    assert data.model != None 
    shutil.rmtree(data.output_location+"/Model_Version2")


    result = runner.invoke(main, ["train", "-tr", training_set, "-m", model, "-o", data.output_location, "-j", True])
    assert os.path.exists(data.output_location+"/Model_Version2")
    assert os.path.isdir(data.output_location+"/Model_Version2")
    for i in os.listdir(data.output_location+"/Model_Version2"):
        if i == "assets" or i == "variables" or i == "Confidence and Accuracy Report" or i == "keras_metadata.pb" or i == "saved_model.pb" or i == "train_data.png" or i == "data.json":
            assert True
        else:
            assert False
    assert os.path.isfile(data.output_location + "/Model_Version2/data.json")
    assert data.model != None 
    # shutil.rmtree(data.output_location+"/Model_Version1")
    # shutil.rmtree(data.output_location+"/Model_Version2")
    pass


# def test_training():
#     #makes a model for testing
#     model = data.model

#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Input(shape=(400, 400, 3)))
#     model.add(tf.keras.layers.Conv2D(32, 3, padding = 'same' , activation='relu'))
#     model.add(MaxPooling2D())
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(units=64, activation='relu'))
#     model.add(tf.keras.layers.Dense(units = 2))

#     model.compile(optimizer = 'adam',
#                   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#                   metrics = ['accuracy'])

#     #save initial weights
#     weights = model.layers[0].get_weights()

#     #data prep
#     DH.data.training_file = "training_set"
#     DH.data.test_file = "testing_set"
#     DH.data.num_epochs = 1
#     train,val = DH.change_input()

#     m.trainModel(train,val)

#     #Save weights after training
#     after_weights = data.model.layers[0].get_weights()

#     #True if before weights and after weights are the same
#     assert not numpy.array_equal(weights,after_weights,equal_nan=False )

# def test_model_creation():
#     model.createModel(2)
#     #data.model initiall equals none but should not after model creation
#     assert data.model != None

# def test_data_saving():
#     runner = CliRunner()
#     result = runner.invoke(main," -m 1 train -tr 2 -t " )
#     print(result.output)

if __name__ == "__main__":
    test_train_command()
    test_train_function()
