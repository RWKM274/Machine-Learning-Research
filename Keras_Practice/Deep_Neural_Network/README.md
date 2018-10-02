# Creating A Denser, Deeper Neural Network

![Deep Neural Network](http://neuralnetworksanddeeplearning.com/images/tikz41.png)

## Creating Our Own Deep Neural Networks in Keras

This time, we used Keras to train our own deep neural networks to identify different datasets that we chose. In this one, we didn't follow a tutorial, but instead used the knowledge that we gained while learning the basics to build one by ourself.

## Datasets

Keras provides various datasets that can be loaded for beginners to try their hand at creating a neural network. They can be loaded like such: 

```Python

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
All of us used datasets from the [Keras's datasets](https://keras.io/datasets/) within the library.

## The Activation Functions Used

### Relu
![Relu Graph](https://www.researchgate.net/profile/Leo_Pauly/publication/319235847/figure/fig3/AS:537056121634820@1505055565670/ReLU-activation-function.png)

The Relu linear activation function is great for hidden layer computation, however it should not be used as a final activation for the output since it can cause skewed results and cause the model to learn very, very slowly.

### Sigmoid
![Sigmoid Graph](https://cdn-images-1.medium.com/max/1600/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

All around great activation function, often used for a variety of Deep Neural Network types due to it's robustness. It can be used as either a hidden layer activation, or it can also be used as the output layer activation.

### Softmax
![Softmax Graph](https://i.stack.imgur.com/ftahr.png)

An activation function that is used as the activation for the final layer of a neural network, and is extremely good at classification problems (identifying one thing from another). Unlike sigmoid though, this function tends to fall short on non-classification problems.

## The Example Programs

* For STOzaki's program, he trained his deep neural network using the [IMDB Dataset](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification).

* For pdemange's program, he trained his deep neural network using the [MNIST Dataset](https://keras.io/datasets/#mnist-database-of-handwritten-digits)

* For 39xdgy's program, he trained his deep neural network also using the [MNIST Dataset](https://keras.io/datasets/#mnist-database-of-handwritten-digits)


