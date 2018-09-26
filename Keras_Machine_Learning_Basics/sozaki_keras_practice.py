from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

# fixed random seed for debugging
np.random.seed(7)

# Creating a neural network with a total of 8 neurons
neuralNetwork = Sequential()
neuralNetwork.add(Dense(4, input_dim=3, activation='sigmoid'))
neuralNetwork.add(Dense(1, activation='sigmoid'))

# Determinting how fast the neural network will learn
optimizing_factor = optimizers.Adam(lr=.01)

# Compile Model
neuralNetwork.compile(loss='mse', optimizer=optimizing_factor, metrics=['accuracy'])

tinput = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
toutput = np.array([1, 1, 1, 1, 1, 0, 1])
# Training the model (fitting)
neuralNetwork.fit(tinput, toutput, epochs=200, batch_size=7)

# evaluate

score = neuralNetwork.evaluate(tinput, toutput)
print("Training samples - \n%s : %.2f%%" % (neuralNetwork.metrics_names[1], score[1]*100))

# Predict with a given input from user and while you do not exit, it will continue to ask the user
while(True):
	try:
		answer0 = input('Give me the first binary number: (type exit to exit this program) ')
		if (answer0 == 'exit'):
			print('exiting')
			break
		answer1 = input('Give me the second binary number: (type exit to exit this program) ')
		if (answer1 == 'exit'):
			print('exiting')
			break
		answer2 = input('Give me the third binary number: (type exit to exit this program) ')
		if (answer2 == 'exit'):
			print('exiting')
			break

		# convert answers to int
		first = int(answer0)
		second = int(answer1)
		third = int(answer2)

		if ((first == 0 or first == 1) and (second == 0 or second == 1) and (third == 0 or third == 1)):
			output = neuralNetwork.predict(np.array([[first, second, third]]))[0][0]
			print ('The Neural Networks answer is: ' + str(output))
		else:
			print('One of the inputs was incorrect. please try again.')
	except (KeyboardInterrupt, SystemExit):
		print('exiting')
		break