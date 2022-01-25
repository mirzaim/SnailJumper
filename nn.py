import numpy as np
from itertools import pairwise


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.n = len(layer_sizes)-1
        self.params = {}

        for i, (inp, outp) in enumerate(pairwise(layer_sizes), start=1):
            self.params[f'W{i}'] = np.random.normal(size=(inp, outp))
            self.params[f'b{i}'] = np.random.normal(size=(outp, ))


    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return np.maximum(x, 0)
        # return np.reciprocal(1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        for i in range(1, self.n):
            x = x @ self.params[f'W{i}'] + self.params[f'b{i}']
            x = self.activation(x)
        x = x @ self.params[f'W{i+1}'] + self.params[f'b{i+1}']
        return x
