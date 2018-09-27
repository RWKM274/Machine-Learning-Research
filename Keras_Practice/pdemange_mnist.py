from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense, Flatten
from keras import optimizers
import numpy as np

class NeuralNet():
	def __init__(self, hLayers, nNPL): 
		self.layers = hLayers
		self.neurons = nNPL
		self.network = self.generateNetwork()

	def generateNetwork(self): 
		model = Sequential()
		model.add(Dense(self.neurons, input_shape=(28,28), activation='sigmoid')) 
		for layer in range(self.layers-1):
			model.add(Dense(self.neurons, activation='sigmoid'))
		model.add(Flatten())
		model.add(Dense(10, activation='softmax'))
		cAdam = optimizers.Adam(lr=.01) 
		model.compile(loss='binary_crossentropy', optimizer=cAdam, metrics=['accuracy'])
		return model

	def train(self, traInputs, traOutputs, nTimes=75): 
		self.network.fit(traInputs, traOutputs, epochs=nTimes, batch_size=100)

	def think(self, testInput): 
		self.network.predict(testInput)

	def accuracy(self, testInputs, testOutputs): 
		scores = self.network.evaluate(testInputs,testOutputs)
		print('Testing with the test dataset of %d images.' % (testInputs))
		print('-'*50)
		print('Testing Dataset Accuracy - %s : %.2f%%' % (self.network.metrics_names[1], scores[1]*100))
		print('-'*50)
def format_outputs(traOutputs):
	arr = np.empty(shape=[0,10])
	for inp in traOutputs: 
		temp = np.zeros((10,))
		temp[inp] = 1
		arr = np.vstack((arr, temp))
	return arr

if __name__ == '__main__': 
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	print('Formatting the output of: y_train')
	print('-'*50)
	y_train = format_outputs(y_train)
	print('Formatting the output of: y_test')
	print('-'*50)
	y_test = format_outputs(y_test)

	net = NeuralNet(2, 150)
	print('-'*50)
	print('Beginning training with MNIST dataset: %d training images' % (len(x_train)))
	print('-'*50)

	net.train(x_train, y_train, 40)

	net.accuracy(x_test, y_test)

