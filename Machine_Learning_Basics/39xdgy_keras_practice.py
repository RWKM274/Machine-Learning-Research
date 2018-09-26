from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

# seed for the random
np.random.seed(7)

# build the layers
model = Sequential()
model.add(Dense(4, input_dim = 3, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# build the network
adam = optimizers.adam(lr = 0.01)
model.compile(loss = 'mse', optimizer = adam, metrics = ['accuracy'])

# building the dataset for training. x is the input and y is the output
x = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]])
y = np.array([0, 1, 1, 1, 1, 1, 1])

# training the dataset
model.fit(x, y, epochs = 1000)

# try to run the test matrix
test_out = model.predict(np.array([[1, 1, 1]]))
print(test_out)

# infinit loop so that users can input the test cases, exit by inputing 6, 6, 6
if __name__ == "__main__":

    while (1 == 1):
        print("type in three numbers for the input, to stop the program, type in 6, 6, 6")
        a = int(input("First number: "))
        b = int(input("Second number: "))
        c = int(input("Three number: "))
        if (a == b and b == c and c == 6):
            break
        input_array = np.array([[a, b, c]])
        print(model.predict(input_array))
