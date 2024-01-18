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

        return self.__A1, self.__A2

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        ''''calculates the gradient descent
          of the neural network and updates
          values of the weights and biases in a single pass'''
        m = X.shape[1]
        dz2 = A2 - Y
        dw2 = np.dot(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
