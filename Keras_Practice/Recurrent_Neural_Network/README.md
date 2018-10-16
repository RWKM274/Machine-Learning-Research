![Recurrent Network](https://cdn-images-1.medium.com/max/2000/0*mRHhGAbsKaJPbT21.png)

# Recurrent Neural Network
This is different from other neural networks because it will remember the output of the previous iteration. A more specfic type of a Recurrent Neural Network(RNN) is call a Long Short-Term Memory(LSTM), which is better than the standard version because it has long term memory.

## What are Recurrent Neural Networks good at? 

![Learning](https://cdn-images-1.medium.com/max/1600/0*SUipu9efyQeKHdlk.)

Recurrent Neural Networks (RNN) are very good at predicting sequences, as well as identifying patterns. Primarily, it's used for natural language processing (ex. Chat bots, text generators, etc.), however it can be used for images when combined with convolutional networks (ex. generating images, generating realistic voices, etc.). 

Here's an example of a LSTM (Long Short Term Model, a modified RNN) learning to talk:

[![Model Learning to Talk](https://i.ytimg.com/vi/FsVSZpoUdSU/maxresdefault.jpg)](https://www.youtube.com/watch?v=FsVSZpoUdSU)

## How are RNNs - LSTMs made in Keras?

You can make a very simple RNN-LSTM in Keras like so:

```Python
from keras.models import Sequential
from keras.layers import LSTM, Dense
...

model = Sequential()
#Create a Recurrent/LSTM layer that takes in a sentence
model.add(LSTM(128, input_shape(len_of_sentences, number_of_unique_characters))
#Create a output Dense layer which is tasked with predicting the next character in the sentence.
model.add(Dense(number_of_unique_characters, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
```

## Links to Tutorials

For more information, read these good articles about Rucurrent Neural Networks:<br/>
[An applied introduction to LSTMs for text generation — using Keras and GPU-enabled Kaggle Kernels](https://medium.freecodecamp.org/applied-introduction-to-lstms-for-text-generation-380158b29fb3)<br/>
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
