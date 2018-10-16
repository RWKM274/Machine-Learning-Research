# Machine-Learning-Research
This repository will serve as a collection of machine learning resources, as well as examples we have created to demonstrate them.

## A Note about Specifications

Computers running Ubuntu were used for this project, and all programs were configured to use Python 3.6. 

If you wish to follow along with our progress as a tutorial, you can download all the python dependencies from the requirements.txt file like so:

```bash
>$ pip install -r requirements.txt
```

## Click Each Link to Track Our Progress (In Order): 

1. [Making a Neural Network by Hand](https://github.com/pdemange/Machine-Learning-Research/tree/master/Machine_Learning_Basics)
2. [Re-Making a Neural Network in Keras](https://github.com/pdemange/Machine-Learning-Research/tree/master/Keras_Machine_Learning_Basics)
3. [Making a Custom Deep Neural Network in Keras](https://github.com/pdemange/Machine-Learning-Research/tree/master/Keras_Practice/Deep_Neural_Network)
4. [Making a Convolutional Neural Network for Image Classification](https://github.com/pdemange/Machine-Learning-Research/tree/master/Keras_Practice/Convolutional_Neural_Network)
5. [Making a Recurrent Neural Network for word processing](https://github.com/pdemange/Machine-Learning-Research/tree/master/Keras_Practice/Recurrent_Neural_Network)
6. [Making Reinforcement Learning Neural Networks](https://github.com/pdemange/Machine-Learning-Research/tree/master/Keras_Practice/Reinforcement_Learning)
7. [Deploying a Neural Network to a Web Service](https://github.com/pdemange/Machine-Learning-Research/tree/master/Flask_Practice)

## Collaborators
* [Ozaki](https://github.com/STOzaki)<br/>
* [39xdgy](https://github.com/39xdgy)

## Downloading images from Google
[hardikvasa](https://github.com/hardikvasa) has an amazing [git repository](https://github.com/hardikvasa/google-images-download) that allows you to download images from Google. Handy to scrap pictures for your neural network.

## Issues
If you get this error that looks something like this: `Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA keras` you need to do this to get rid of the warning `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'`.<br/>
[For more information visite this stackoverflow page](https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u)

## Installation
1. `git clone https://github.com/pdemange/Machine-Learning-Research.git`<br/>
2. install all necessary packages using pip3: `pip3 install -r requirement.txt`
