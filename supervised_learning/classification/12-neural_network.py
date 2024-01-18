#!/usr/bin/env python3
'''Defines a Neural Network with one hidden
layer performing binary classification'''

import numpy as np


class NeuralNetwork:
    '''class Neural Network defining a neural net with one hidden layer'''

    def __init__(self, nx, nodes):
        '''class constructor'''
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''getter function for W1'''
        return self.__W1

    @property
    def b1(self):
        '''getter function for b1'''
        return self.__b1

    @property
    def A1(self):
        '''getter function for A1'''
        return self.__A1

    @property
    def W2(self):
        '''getter function for W2'''
        return self.__W2

    @property
    def b2(self):
        '''getter function for b2'''
        return self.__b2

    @property
    def A2(self):
        '''getter function for A2'''
        return self.__A2

    def forward_prop(self, X):
        '''calculates the forward propagation of the neural network'''

        '''multiply transposed weights with input vector'''
        weighted_sum1 = np.dot(self.W1, X)
        ''' Add the bias to the dot'''
        Z1 = np.add(weighted_sum1, self.b1)
        '''apply the activation function, sigmoid
        to get the output for the hidden layer'''

        self.__A1 = 1/(1+np.exp(-Z1))

        ''''output from the hidden layer will be
        the input for the output neuron'''

        weighted_sum_2 = np.dot(self.W2, self.__A1)
        Z2 = np.add(weighted_sum_2, self.b2)
        self.__A2 = 1/(1+np.exp(-Z2))

        return self.__A2

    def cost(self, Y, A):
        '''cost function for our Neuron'''

        m = A.shape[1]
        '''use binary cross Entropy function to calculate logloss'''
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        total_loss = np.sum(loss)

        '''divive the loss by the num of examples'''

        cost_func = total_loss/m

        return cost_func

    def evaluate(self, X, Y):
        '''evaluates the predictions of the neural net'''
        # first perform forward propagation
        Y_pred = self.forward_prop(X)
        # then calculate the cost of running the net
        cost = self.cost(Y, Y_pred)
        # change the values
        y_convert = np.where(Y_pred >= 0.5, 1, 0)
        return y_convert, cost
