import numpy as np

""" Takes in the number of neurons and the number of inputs
	you would like and build a layer for you.
	Each layer will have a input from the previous layer and
	its neurons and its own neurons
"""
class NeuronLayer():
    def __init__(self, numberOfNeurons, numberOfInputsPerNeuron):
        self.synaptic_weights = 2 * np.random.random((numberOfInputsPerNeuron, numberOfNeurons)) - 1


# Class that initializes with two layers that are the NeuronLayer class
class NeuralNetwork():
	def __init__(self, layer1, layer2):
		self.layer1 = layer1
		self.layer2 = layer2


	# takes in a number and normalizIndicates how confident we are about the existing weightes the number between 0 and 1
	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))


	# Gives the amount of adjustment we need to give to the weights
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	""" We are training the neural network, by taking in the training set,
		then expected output set, and how many times we want the nerual
		network to iterate through the training (epoches)
	"""
	def train(self, trainingSetInputs, trainingSetOutputs, numberOfTrainingIterations):
		for iteration in range(numberOfTrainingIterations):

			# train the nerual network with the training set
			outputFromLayer1, outputFromLayer2 = self.think(trainingSetIputs)

			# Calculate the error for layer 2
			layer2Error = trainingSetOutputs - outputFromLayer2
			layer2Delta = layer2Error * self.__sigmoid_derivative(outputFromLayer2)

			# Calculate the error for layer 1
			layer1Error = layer2Delta.dot(self.layer2.synaptic_weights.T)
			layer1Delta = layer1Error * self.__sigmoid_derivative(outputFromLayer1)

			# Calculate how much you need to adjust each weights by in each layer
			layer1Adjustment = trainingSetIputs.T.dot(layer1Delta)
			layer2Adjustment = outputFromLayer1.T.dot(layer2Delta)

			# Adjust the weights in each layer
			self.layer1.synaptic_weights += layer1Adjustment
			self.layer2.synaptic_weights += layer2Adjustment



	# Using an input, it 'thinks' about what the outcome will be
	def think(self, inputs):
		outputFromLayer1 = self.__sigmoid(np.dot(inputs, self.layer1.synaptic_weights))
		outputFromLayer2 = self.__sigmoid(np.dot(outputFromLayer1, self.layer2.synaptic_weights))
		return outputFromLayer1, outputFromLayer2

	# Prints out all of the neural network weights
	def print_weights(self):
		print ('Layer 1 (4 neruons, each with 3 inputs): ')
		print (self.layer1.synaptic_weights)
		print ('Layer 2 (1 neruon, with 4 inputs):')
		print (self.layer2.synaptic_weights)

	""" steping: if input hits our threshold (0.5), then it fires an output,
		otherwise it does not do anything
	"""
	def step(input):
		if(input >= 0.5):
			input = 1
		else:
			input = 0
		return input


""" Main: creates a random seed and 
"""
if __name__ == "__main__":

	# Seeding to enable us to go back and debug
	np.random.seed(1054)

	# Creating layer 1 (4 neurons, 3 inputs)
	layer1 = NeuronLayer(4, 3)

	# Creating layer 2 (1 neuron, 4 inputs)
	layer2 = NeuronLayer(1, 4)

	# Create neral network
	neuralNetwork = NeuralNetwork(layer1, layer2)

	# prints all of the stages for easy read
	# Stage 1: Prints all the randomizes weights
	print ('Stage 1: Randomizing synaptic weights: ')
	neuralNetwork.print_weights()

	# Training set: 7 examples, each with 3 inputs and 1 output
	trainingSetIputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
	trainingSetOutputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

	# Train neural network with the training set, doing that 60,000 times
	neuralNetwork.train(trainingSetIputs, trainingSetOutputs, 60000)

	# Stage 2: Weights after training
	print ('Stage 2: New synaptic weights after training: ')
	neuralNetwork.print_weights()

	# Stage 3: Test the new neural network
	print ('Stage 3: Consider a new situation [1, 1, 0] = ?')
	hidden_state, output = neuralNetwork.think(np.array([1, 1, 0]))
	reluOutput = neuralNetwork.relu(output)
	print (output)
