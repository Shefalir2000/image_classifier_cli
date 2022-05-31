import numpy
from click.testing import CliRunner
from hello.src import there_command
from hello.__main__ import main
import pytest
from hello.src import model as m
import tensorflow as tf
from keras.layers import MaxPooling2D
import tensorflow_datasets as tfds
from hello.src import DataHandler as DH
from hello.src import DataClass
from hello.src import model

data = DataClass.Parameters()

def test_train_function():
    print("function train test")
    

def test_train_command():
    print("command train test")