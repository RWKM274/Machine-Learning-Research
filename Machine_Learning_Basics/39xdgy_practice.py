import numpy as np

# this class creates the layers inside of the NeuralNetwork
class NeuronLayers():
    
    # the inital, take two input of how many input neurons and how many output neurons
    def __init__(self, number_of_neurons, number_of_inputs):
        self.weight = 2 * np.random.random((number_of_inputs, number_of_neurons)) - 1
        
# this class creates the whole Neural Network
class NeuralNetwork():
    
    # the inital, take two input of the two layers, no outputs
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # this function take the output and normalizes it
    '''
    input: one float
    output: one float
    '''
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # rate of change about how mach different of the error, out put one float
    '''
    input: one float
    output: one float
    '''
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # actually training the network, would update the weight inside the layers
    '''
    input: three things, training inputs array, training output array, the e_poch: one int that shows how many times the set would be trained
    output: None
    '''
    def train(self, training_set_inputs, training_set_outputs, e_poch):
        for i in range(e_poch):
            
            output_from_layer1, output_from_layer2 = self.think(training_set_inputs)

            layer2_error = training_set_outputs - output_from_layer2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer2)

            layer1_error = layer2_delta.dot(self.layer2.weight.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer1)

            layer1_ajustment = training_set_inputs.T.dot(layer1_delta)
            layer2_ajustment = output_from_layer1.T.dot(layer2_delta)

            self.layer1.weight += layer1_ajustment
            self.layer2.weight += layer2_ajustment


    # actually try to predect what is the answer
    '''
    input: the input array 
    output: two output layers, and the second one is the answer
    '''
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(np.dot(inputs, self.layer1.weight))
        output_from_layer2 = self.__sigmoid(np.dot(output_from_layer1, self.layer2.weight))
        return output_from_layer1, output_from_layer2

    # print function that would print out the weight
    '''
    input: None
    output: None
    '''
    def print_weights(self):
        print ("layer 1")
        print (self.layer1.weight)
        print ("layer 2")
        print (self.layer2.weight)

# main test function
if __name__ == "__main__":
    # set the random seed
    np.random.seed(1)

    # two layers that take 3 input to 4 neurons that leads to 1 output neuron
    layer1 = NeuronLayers(4, 3)
    layer2 = NeuronLayers(1, 4)

    # create the Neural Network
    network = NeuralNetwork(layer1, layer2)

    # show the start up weight in the network
    print("The start up weight in the network")
    network.print_weights()

    # create the inputs and outputs for training
    # this traning sets are going to teach the neural network to learn how to do (or ||)
    train_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    train_set_outputs = np.array([[1, 1, 1, 1, 1, 1, 0]]).T

    # train the network
    network.train(train_set_inputs, train_set_outputs, 120000)

    # print the weight after training
    print("The weight after the training")
    network.print_weights()

    # think, and print the test case which is 1, 1, 0 and the output of it
    print("Test to see if the model are good enough by inputing [1, 1, 0]")
    hidden_state, output = network.think(np.array([1, 1, 0]))
    print(output)

    # infinit loop for testing, free input and to check if the output is correct or not. 
    while(1 == 1):
        print("For testing, please type in three numbers, to end the test, type in 6, 6, 6")
        a = int(input("First number: "))
        b = int(input("Second number: "))
        c = int(input("Thrid number: "))

        if(a == b and b == c and c == 6):
            break

        hidden_state, output = network.think(np.array([a, b, c]))
        print(output)
