# Image Classifier: A simple tool to create and test models

##Requirements
Requires Python version >= 3.10 to run setup.py

## Summary
ImgClass is a simple CLI tool that meant to help get users started with creating and messing around with ML models. It showcases the use of Click and logging for tool development through multiple files. It also includes everything needed to deliver and install a pypi compatable project. This tool also includes the basic infrastructure needed to produce a product that can classify images, when given a training data set. Below you will find the details of how to install and use the tool to its full capability. 
I suggest reading through the comments of the project and reading about the packages used throughout the project fpr a better understanding of how packeges were used throughout the tool. 

### Installation

#### Python

When running this program in python make sure that you reinstall the packages after all the changes you make and before each run. If you do not run this command your changes will not be saved. 
The function call you must use is ```pip install .```


### Usage

ImgClass was built using Click for a better CLI. Each "argument" is defined as an option for ease of use, however, keep in mind some options are required. 

To see all available commands and options

``` ImgClass --help ```

## Command: train

``` ImgClass train -tr "location of training data" [options]```

To see all options and explainations:

``` ImgClass train --help```

In the following example a user would connect their complete path to the data after the -tr flag and would include any other flags they would deam appropriate. The tool would then train and save a new or existing model to the designated output folder.

#### Example:

``` ImgClass train -tr "C:\Program Files\Training_data" ```

Output:

```
2022-06-01 11:30:45.938158: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Found 500 files belonging to 5 classes.
Using 100 files for validation.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 rescaling (Rescaling)       (None, 400, 400, 3)       0

 conv2d (Conv2D)             (None, 400, 400, 16)      448

 max_pooling2d (MaxPooling2D  (None, 200, 200, 16)     0
 )

 conv2d_1 (Conv2D)           (None, 200, 200, 32)      4640

 max_pooling2d_1 (MaxPooling  (None, 100, 100, 32)     0
 2D)

 flatten (Flatten)           (None, 320000)            0

 dropout (Dropout)           (None, 320000)            0

 dense (Dense)               (None, 64)                20480064

 dropout_1 (Dropout)         (None, 64)                0

 dense_1 (Dense)             (None, 64)                4160

 dense_2 (Dense)             (None, 5)                 325

=================================================================
Total params: 20,489,637
Trainable params: 20,489,637
Non-trainable params: 0
_________________________________________________________________
13/13 [==============================] - 16s 1s/step - loss: 11.4800 - accuracy: 0.1550 - val_loss: 1.6728 - val_accuracy: 0.2700
INFO:tensorflow:Assets written to: C:\Users\sranjan31\Source\clitoolpleasework\image_classifier_cli/Output/Model_Version1\assets
Train
```
#### Options:

  ```-m, --model TEXT```                If a model exists then use the model that
                                  	will be trained.

  ```-tr, --training TEXT```            Adds the data that the model will be trained
                                  	on.  [required]

  ```-e, --epochs INTEGER```            Changes the number of epochs that will be
                                  	done during training.

  ```-b, --batch INTEGER```             Changes the batch number that will be used
                                  	during training.

  ```-h, --height INTEGER```            Changes the height of the images during
                                  	training.

  ```-w, --width INTEGER```             Changes the width of the images during
                                  	training.

  ```-o, --output TEXT```               Changes where the file will be created to
                                  	store analysis about the model creation
                                  	process. If no input, will generate an "Output" file in 
                                   the root directory 

  ```--help```                          Show this message and exit.

## Command: predict

``` ImgClass predict -te "location of testing data" -m "location of the model" [options]```

To see all options and explainations:

``` ImgClass predict --help```

In the following example a user would connect their complete path to the data after the -tr flag and would include any other flags they would deam appropriate.

#### Example:

``` ImgClass predict -te "C:\Program Files\Testing_data" -m "C:\Program Files\model" ```

Output:

```
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
['AK', 'Ala_Idris', 'Buzgulu', 'Dimnit']
list C:\Users\sranjan31\PycharmProjects\imgClassifierTest\gtriScreenClassification\testing_set\Grapevine
sub AK
Ak (1).png
1/1 [==============================] - 0s 172ms/step
[[-3.5455697 -3.6295052 -4.079938  -2.8049839 -4.71748  ]]
This image most likely belongs to Dimnit with a 42.69 percent confidence.
tf.Tensor([0.20356365 0.18717486 0.11929631 0.4269063  0.06305884], shape=(5,), dtype=float32)
Ak (2).png
1/1 [==============================] - 0s 47ms/step
[[-2.8605902 -2.693569  -2.986797  -2.099851  -3.579639 ]]
This image most likely belongs to Dimnit with a 37.61 percent confidence.
tf.Tensor([0.1757381  0.20768368 0.15490127 0.37605453 0.08562233], shape=(5,), dtype=float32)
Ak (3).png
1/1 [==============================] - 0s 63ms/step
[[-2.652342  -2.44133   -2.895125  -1.9432619 -3.2940793]]
This image most likely belongs to Dimnit with a 36.43 percent confidence.
tf.Tensor([0.17927998 0.22139776 0.14063472 0.36431867 0.0943689 ], shape=(5,), dtype=float32)
Ak (4).png
1/1 [==============================] - 0s 63ms/step
[[-2.916375  -2.7565088 -2.9098601 -2.1177814 -3.6556222]]
This image most likely belongs to Dimnit with a 37.80 percent confidence.
tf.Tensor([0.17007451 0.19955756 0.17118612 0.3779758  0.08120601], shape=(5,), dtype=float32)
Ak (5).png
1/1 [==============================] - 0s 47ms/step
[[-2.8134582 -2.6721783 -3.0457902 -2.1349456 -3.554397 ]]
This image most likely belongs to Dimnit with a 36.55 percent confidence.
tf.Tensor([0.18545856 0.21360134 0.14700983 0.3655284  0.08840182], shape=(5,), dtype=float32)
```
#### Options:

  ```-m, --model TEXT```                Conducts the predictions using the model that is passed in.  [required]

  ```-te, --testing TEXT```             Adds the data that the model will be tested
                                  	on.  [required]

  ```-e, --epochs INTEGER```            Changes the number of epochs that will be
                                  	done during training.*

  ```-b, --batch INTEGER```             Changes the batch number that will be used
                                  	during training.*

  ```-h, --height INTEGER```            Changes the height of the images during
                                  	training.*

  ```-ct, --confidence_threshold INTEGER```
                                  	Changes the height of the images during
                                  	training.*

  ```-w, --width INTEGER```             Changes the width of the images during
                                  	training.*

  ```-o, --output TEXT```               Changes where the file will be created to
                                  	store analysis about the model creation
                                  	process.*

  ```--nr```				Decide whether the report is generated or
                                  	not. If nothing is entered the report will
                                  	be generated.

  ```--help```                          Show this message and exit.








