from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers
#data for mnist, the handwitten number pictures
from keras.datasets import mnist

import numpy as np


# each input is 28*28
# total training set is 60,000
# total testing set is 10,000
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# the y_train and y_test is a single array that have the right number
# So I have to transfor into a double array that each sub array are 10 elements with 0 and 1 
real_y_train = np.empty(shape = [0, 10])
real_y_test = np.empty(shape = [0, 10])


# build up the training set output
for i in y_train:
    trans = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    trans[i] = 1
    trans = np.array([trans])
    real_y_train = np.vstack((real_y_train, trans))


# build up the testing set output
for i in y_test:
    trans = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    trans[i] = 1
    trans = np.array([trans])
    real_y_test = np.vstack((real_y_test, trans))

    
np.random.seed(7)

# create the model of the network
model = Sequential()
model.add(Dense(100, input_shape = (28, 28), activation = 'sigmoid'))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

# build the whole model
adam = optimizers.adam(lr = 0.01)
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

# train the network
model.fit(x_train, real_y_train, epochs = 3

# checking the output by evaluate the testing set. 
score = model.evaluate(x_test, real_y_test)
print("This is the final output!!!")
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
