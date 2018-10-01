# Making a Neural Network

![Dense Neural Network](https://i.stack.imgur.com/iHW2o.jpg)

### The Building

A neural network is composed of many different layers of "neurons", with all the layers being interconnected to eachother by their "synaptic weights". Both the neurons and the synaptic weights have values assigned to them in order for the network to perform a feed-forward calculation, or to be able to "think". 

### How It "Thinks"
The calculation used for the feed-forward process is such:

![Equation](https://www.learnopencv.com/wp-content/uploads/2017/10/neuron-diagram.jpg)

For each neuron in the network, the connected input neurons and their weights are multiplied and then summed with the weight of the neuron that is "firing". After this, the value is then piped into an "activation" function, which determines the value of the neuron as whether it "fired" or not.

This happens for each neuron in a layer, and for every layer. The output values of the the final layer is what the neural network has "thought". From this output, we can interpret the numbers as answers to the problem we gave it, which can be anything we deem it can learn to solve with the proper input data.

## How It "Learns"

After the feed forward process, our network might not be that good, or it might've not given us the correct answer at all. So, we would need to make it "learn" to become better at it's calculations. In order to do this, we need to feed it data that we already know the answer to and watch it learn to determine it's accuracy (this is supervised learning).

![gradient](https://cdn-images-1.medium.com/max/1600/1*6sDUTAbKX_ICVVAjunCo3g.png)

From the inputted data, we compare the output that the network predicts with the answer we know is correct. If it's wrong, we calculate by how much it's wrong through the gradient of a loss function. Within our programs, we use the sigmoid derivative function to determine this.

![Sigmoid Derivative](https://cdn-images-1.medium.com/max/1600/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

After we've determined by how much we need to adjust it by using the sigmoid derivative, we adjust each of the layers by calculated amount in order to have it "learn" the correct values. After a series of training sessions, the network will become better and better at doing the task we've assigned it, as well as be able to give us accurate answers to the problems we haven't introduced it to!


## The Example Programs

Within this directory are our example programs that we each made my hand (using only numpy) to demonstrate this entire process. STOzaki made a network that will calculate NAND gate problems, pdemange made one to calculate AND problems, and 39xdgy made one to calculate OR problems. All you need to run the programs are Numpy for Python 3.

## Sources

We followed [this](https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a) article while building our networks to help us understand what we were doing.
