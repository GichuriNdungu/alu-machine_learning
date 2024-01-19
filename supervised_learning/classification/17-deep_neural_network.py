#!/usr/bin/env python3
'''Defines a binary classification deep neural net'''
import numpy as np


class DeepNeuralNetwork:
    '''defines a deep neural network
    performing binary classification:'''

    def __init__(self, nx, layers):
        '''class constructor'''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if min(layers) < 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        # Initialize the first layer using the number of input features (nx)
        self.__weights['W1'] = np.random.randn(layers[0], nx) * np.sqrt(2/nx)
        self.__weights['b1'] = np.zeros((layers[0], 1))

        # Initialize the rest of the layers
        for layer in range(1, self.L):
            self.__weights['W' + str(layer+1)] = np.random.randn(layers[layer],
                                                                 layers[layer-1]) * np.sqrt(2/layers[layer-1])
            self.__weights['b' + str(layer+1)] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        ''''getter func for L'''
        return self.__L

    @property
    def cache(self):
        '''getter func for cache'''
        return self.__cache

    @property
    def weights(self):
        ''''getter func for weights and biases'''
        return self.__weights
