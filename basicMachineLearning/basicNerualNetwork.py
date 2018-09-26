import numpy as np
class NeuronLayer():
    def __init__(self, numberOfNeurons, numberOfInputsPerNeuron):
        self.synaptic_weights = 2 * np.random.random((numberOfInputsPerNeuron, numberOfNeurons)) - 1


class NeuralNetwork():
	def __init__(self, layer1, layer2):
		self.layer1 = layer1
		self.layer2 = layer2


	# normalise input between 0 and 1
	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))


	# Indicates how confident we are about the existing weight
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	# We training the neural network
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



	# thinking
	def think(self, inputs):
		outputFromLayer1 = self.__sigmoid(np.dot(inputs, self.layer1.synaptic_weights))
		outputFromLayer2 = self.__sigmoid(np.dot(outputFromLayer1, self.layer2.synaptic_weights))
		return outputFromLayer1, outputFromLayer2

	# The neural network weights
	def print_weights(self):
		print ('		Layer 1 (4 neruons, each with 3 inputs): ')
		print (self.layer1.synaptic_weights)
		print (' 	Layer 2 (1 neruon, with 4 inputs):')
		print (self.layer2.synaptic_weights)

if __name__ == "__main__":

	#Seed
	np.random.seed(1)

	# Creating layer 1 (4 neurons, 3 inputs)
	layer1 = NeuronLayer(4, 3)

	# Creating layer 2 (1 neuron, 4 inputs)
	layer2 = NeuronLayer(1, 4)

	# Create neral network
	neuralNetwork = NeuralNetwork(layer1, layer2)

	print ('Stage 1: Randomizing synaptic weights: ')
	neuralNetwork.print_weights()

	# Training set: 7 examples, each with 3 inputs and 1 output
	trainingSetIputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
	trainingSetOutputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

	# Train neural network with the training set, doing that 60,000 times
	neuralNetwork.train(trainingSetIputs, trainingSetOutputs, 60000)

	print ('Stage 2: New synaptic weights after training: ')
	neuralNetwork.print_weights()

	# Test the new neural network
	print ('Stage 3: Consider a new situation [1, 1, 0] = ?')
	hidden_state, output = neuralNetwork.think(np.array([1, 1, 0]))
	print (output)
