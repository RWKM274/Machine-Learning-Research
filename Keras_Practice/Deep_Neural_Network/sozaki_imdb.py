from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb
from keras import optimizers
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
import numpy as np

# fixed random seed for debugging
np.random.seed(7)

top_words = 5000
max_words = 500

# grabing dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=top_words,
                                                      skip_top=0,
                                                      maxlen=max_words,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

	


X_train = sequence.pad_sequences(x_train, maxlen=max_words)
X_test = sequence.pad_sequences(x_test, maxlen=max_words)

# Creating a neural network
neuralNetwork = Sequential()
neuralNetwork.add(Embedding(top_words, 32, input_length=max_words))
neuralNetwork.add(Dense(50, activation='relu'))
neuralNetwork.add(Dense(50, activation='relu'))
neuralNetwork.add(Flatten())
neuralNetwork.add(Dense(1, activation='sigmoid'))

# # Determinting how fast the neural network will learn
optimizing_factor = optimizers.Adam(lr=.01)

# Compile Model
neuralNetwork.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Training the model (fitting)
# neuralNetwork.fit(np.array(x_train, dtype=int), np.array(y_train, dtype=int), epochs=200, batch_size=32)

neuralNetwork.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=1)


# Final evaluation of the model
scores = neuralNetwork.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))