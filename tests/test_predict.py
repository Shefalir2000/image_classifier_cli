import numpy
from click.testing import CliRunner
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

def test_predict_function():
    print("function predict test")

def test_predict_command():
    print("command predict test")
