from keras.models import Sequential
from keras.layers import Dense 
from keras import optimizers 
import numpy as np
import sys

class ClassifierKeras():
	def __init__(self):
		self.classifier = self.generateModel()

	#Compress and return a model composed of our neurons
	def generateModel(self):
		model = Sequential()
		model.add(Dense(10, input_dim=4, activation='relu'))
		model.add(Dense(5, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		adamBoi = optimizers.Adam(lr=.01)
		model.compile(loss='mse', optimizer=adamBoi, metrics=['accuracy'])
		return model

	#Train the compressed model
	def train(self, traInputs, traOutputs, nTimes):
		self.classifier.fit(traInputs, traOutputs, epochs=nTimes, batch_size=4)

	#Make a prediction based on inputs
	def think(self, Inboi): 
		return self.classifier.predict(Inboi)

	#Get statistics
	def evaluate(self, traInputs, traOutputs):
		scores = self.classifier.evaluate(traInputs, traOutputs)
		print('-'*50)
		print("\n Training Samples - %s: %.2f%%" % (self.classifier.metrics_names[1], scores[1]*100))
		print('-'*50)

if __name__ == '__main__':
	#Make the network
	net = ClassifierKeras()
	#Initialize random seed
	np.random.seed(1337)
	#Get inputs and outputs
	tIn = np.array([[0,1,0,1],[1,0,0,0],[1,1,0,0],[1,1,1,1],[1,0,0,1],[0,0,0,0],[1,1,1,0]])
	tOut = np.array([0,0,0,1,0,0,0])
	#Train and evaluate the network
	net.train(tIn, tOut, 50)
	net.evaluate(tIn, tOut)
	#
	while True: 
		try:
			first = int(input('Enter first number to AND: '))
			second = int(input('Enter second number to AND: '))
			third = int(input('Enter third number to AND: '))
			fourth = int(input('Enter fourth number to AND: '))
			output = net.think(np.array([[first, second, third, fourth]]))
			print('Confidence: '+str(output[0][0]))
			print('-'*50)
		except:
			sys.exit(0)