# Hello: A simple, first python project template

## Summary
Hello is a simple CLI tool meant to get users started with a deliverable project. It showcases the use of [Click](https://click.palletsprojects.com/en/8.1.x/) and a glimpse into [logging](https://docs.python.org/3/howto/logging.html) for "proper" tool development. It also includes everything needed to deliver and install a pypi compatible project, as well as the infrastructure needed to product a portable container for the project. The details below dive more into how to use this project as well as how to install and see it in action!

I first suggest installing and playing with the CLI if you aren't familiar with that process. If you are, skip to [Make it My Own](#make-it-my-own) section to see what you can do to start creating your own tool.

Finally, use this readme as a guide to writing a readme for your next project!

### Installation

#### Container
You can run everything using the latest container release:

First pull the container:

```podman pull [image]```

Then run the command using the newly pulled container:

```podman run [image] hello COMMAND [options]```

#### Python
Hello can also be installed locally as a python CLI tool:

```python setup.py install```

## Usage
Hello was built using Click for a better CLI user experience. Each "argument" is defined as an option for ease of use, but note that some of the options are required (see Options below).

To see all available commands and options:

```hello --help```

### Command: there

```hello there [options]```

For help:

```hello there --help```

In the following example a user would connect to their instance of Label Studio using their host path and API token _You can find your user token on the User Account page in Label Studio_. They are then specifying the project id and the VOC export type (VOC being used generally for Tensorflow object detection projects):

#### Example:

```
hello there -n Austin -g
```

output: Hello there Austin, how are you?

#### Options:

`--name, -n [String]`: Name that you would like to include in the greeting. [required]

`--greeting, -g`: Adds "how are you?" to the greeting.


## Contributing

Thanks for considering, we need your contributions to help this project come to fruition.

Here are some important resources:

- Bugs? [Issues](https://github.com/rutheferd/hello_template/issues) is where to report them!
- Please utilize pre-commits or manually invoke black and flake8 to ensure style compliance.

## Make it My Own

1. First things first, change the `/hello` directory to whatever you'd like the name of the project to be!

2. Write some tests! You can add python test files to `/tests` see [pytest](https://docs.pytest.org/en/7.1.x/) for more information on pytest for testing. Also check out `tests/test_there.py` for a simple example.

3. Add you python file(s) under `hello/src`. For greater compliance with click, check `hello/src/there_command.py` for an exmaple.

4. For argument handling we are using [Click](https://click.palletsprojects.com/en/8.1.x/), and an simple example is available in `hello/__main__.py`. Note that you will need to import your code from source to run in the `__main__.py` file. See the imports at the top for an example.

5. In the `requirements.txt` update with all of the packages that your code will expect to have.

6. Now you need to modify the `setup.py` file:
    - Update `name` to your new project name
    - Update `description` to something compelling for your project.
    - Update `python_requires` with the mininum required python version for your project.
    - Update `entry_points` with the name of the folder updated in step one, see the exmaple for better understanding.
    - Update the `author, keywords, license, url, author_email` with the relevant information.

And that should be it! Congratulations, you should now be able to isntall your project and run it as a CLI tool. 
**Note** that this is a simplified guide to how you can setup your project, and is really the bare minimum to get you going. I will leave it to you to leave an issue asking further questions, or do some digging and contribute what you've learned to the project and this guide!

🚧 Gitlab implementation coming soon! 🚧

## License

   Copyright 2022 Austin Ruth

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


# Image Classifier: A simple tool to create and test models

## Summary
ImgClass is a simple CLI tool that meant to helpget users started with creating and messing around with ML models. It showcases the use of Click and logging for tool development through mu;tiple files. It also includes everything needed to deliver and install a pypi compatable project. This tool also includes the basic infrastructure needed to produce a product that can classify images, when given a training data set. Below you will find the details of how to install and use the tool to its full capability. 
I suggest reading through the comments of the project and reading about the packages used throughout the project fpr a better understanding of how packeges were used throughout the tool. 

### Installation

#### Python

When running this program in python make sure that you reinstall the packages after all the chnages you make adn before each run. If you do not run this command your changes will not be saved. 
The function call you must use is ```pip install .```


### Usage

ImgClass was built using Click for a better CLI. Each "argument" is defined as an option for ease of use, however, keep in mind some options are required. 

To see all available commands and options

``` ImgClass --help ```

## Command: train

``` ImgClass train -tr "location of training data" [options]```

To see all options and explainations:

``` ImgClass train --help```

In the following example a user would connect their complete path to the data after the -tr flag and would include any other flags they would deam appropriate.

#### Example:

``` ImgClass train -tr "C:\Program Files\Training_data" ```

output:

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
                                  	will be tested.

  ```-tr, --training TEXT```            Adds the data that the model will be trained
                                  	on.  [required]

  ```-e, --epochs INTEGER```            Changes the number of epochs that will be
                                  	done during training.

  ```-b, --batch INTEGER```             Changes the batch number that will be used
                                  	during training.

  ```-h, --height INTEGER```            Changes the height of the images during
                                  	training.

  ```-ct, --confidence_threshold INTEGER```
                                  	Changes the height of the images during
                                  	training.

  ```-w, --width INTEGER```             Changes the width of the images during
                                  	training.

  ```-o, --output TEXT```               Changes where the file will be created to
                                  	store analysis about the model creation
                                  	process.

  --help                          	Show this message and exit.

#### Options:

  ```-m, --model TEXT```                If a model exists then use the model that
                                  	will be tested.  [required]

  ```-te, --testing TEXT```             Adds the data that the model will be tested
                                  	on.  [required]

  ```-e, --epochs INTEGER```            Changes the number of epochs that will be
                                  	done during training.

  ```-b, --batch INTEGER```             Changes the batch number that will be used
                                  	during training.

  ```-h, --height INTEGER```            Changes the height of the images during
                                  	training.

  ```-ct, --confidence_threshold INTEGER```
                                  	Changes the height of the images during
                                  	training.

  ```-w, --width INTEGER```             Changes the width of the images during
                                  	training.

  ```-o, --output TEXT```               Changes where the file will be created to
                                  	store analysis about the model creation
                                  	process.

  ```--nr```				Decide whether the report is generated or
                                  	not. If nothing is entered the report will
                                  	be generated.

  ```--help```                          Show this message and exit.








