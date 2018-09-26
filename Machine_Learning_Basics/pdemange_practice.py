import numpy as np
import sys

class NeuronLayer():
	#Building the synaptic weights of a layer using randomization
	def __init__(self,numberOfNeurons, numberOfInputsPerNeuron):
		self.weights = 2 * np.random.random((numberOfInputsPerNeuron, numberOfNeurons)) -1

class NeuralNetwork():

	#Initialization function, "compressing" the layers for the network
	def __init__(self, layer1, layer2):
		self.l1 = layer1
		self.l2 = layer2 

	'''
	The activation function for our neuron layers

	Inputs:
	------------------------
	x - The dot product of the synaptic weights and the layer input
	------------------------

	Returns: 
	------------------------
	The output for the given layer
	------------------------

	This function is used to calculate what the network "thinks" for
	a given layer's input
	'''
	def sigmoid(self, x): 
		return 1/(1+np.exp(-x))

	'''
	Get the gradient of an output: derivative

	Inputs:
	------------------------
	x - The output of a given layer
	------------------------

	Returns: 

	------------------------
	The gradient of the given layer output
	------------------------

	This function is the sigmoid derivative, and 
	is used to calculate the adjustment for a layer.
	'''

	def derivative(self, x):
		return x * (1-x)

	'''
	Training function: train

	Inputs: 
	-------------------------
	traInputs - Training data

	traOutputs - Correct outputs for the training data

	nTimes - Number of epochs to train the network
	-------------------------

	Train the network by getting the loss, calculating the gradients,
	and adjusting the weights for each layer (layers 1 and 2).
	'''

	def train(self, traInputs, traOutputs, nTimes): 
		for aTime in range(nTimes):
			l1Out, l2Out = self.think(traInputs) 

			l2Error = traOutputs - l2Out
			l2Delta = l2Error * self.derivative(l2Out)

			l1Error = l2Delta.dot(self.l2.weights.T)
			l1Delta = l1Error * self.derivative(l1Out)

			adjust1 = traInputs.T.dot(l1Delta)
			adjust2 = l2Out.T.dot(l2Delta) 

			self.l1.weights += adjust1
			self.l2.weights += adjust2

	'''
	Thinking or Prediction function: think

	Inputs:
	-------------------------
	inputs - The inputs for the first layer (the "input layer")
	-------------------------

	Returns: 
	-------------------------
	out1 - The output calculation of the first layer

	out2 - The output calculation of the second layer
	-------------------------

	This function will make a "prediction" based on the current
	synaptic weights within each of the layers.

	'''
	def think(self, inputs):
		out1 = self.sigmoid(np.dot(inputs, self.l1.weights))
		out2 = self.sigmoid(np.dot(out1, self.l2.weights))
		return out1, out2

	'''
	Printing function: printWeights

	Prints out the current synaptic weights for each of the layers
	'''
	def printWeights(self):
		print('Layer 1:')
		print(self.l1.weights)
		print('Layer 2:')
		print(self.l2.weights)

'''
Main function
-------------
'''

if __name__ == '__main__':

	np.random.seed(1337)

	layer1 = NeuronLayer(4,2)
	layer2 = NeuronLayer(1,4) 
	net = NeuralNetwork(layer1, layer2)

	#Print the initial random weights
	print('Stage 1 - Weight Dump:')
	print('-'*50)
	net.printWeights()
	print('-'*50)
	print('\n')

	#Train the network to perform AND operations on two "bits"
	tIn = np.array([[0,1],[1,0],[1,1],[0,0]])
	tOut = np.array([[0,0,1,0]]).T

	#Train the neural network for 600000 epochs
	net.train(tIn, tOut, 600000)

	#Print the trained weights
	print('Stage 2 - New Weights:')
	print('-'*50)
	net.printWeights()
	print('-'*50)
	print('\n')

	#Start an interactive prediction interface
	print('Stage 3 - Predicting:')
	print('-'*50)
	while True:
		try:
			first = int(input('Enter first number to AND: '))
			second = int(input('Enter second number to AND: '))

			_, output = net.think(np.array([first,second]))
			'''
			Do a "step" calculation on the output, put the output
			between a 0 and 1 like a real AND output.
			'''
			if output >= .5:
				output = 1 
			else: 
				output = 0
			print('Confidence: '+str(output))
			print('-'*50)
		except: 
			sys.exit(0)



