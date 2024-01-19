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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for layer in range(1, self.L):
            self.weights['W' + str(layer)] = np.random.randn(layers[0],
                                                             nx) * np.sqrt(2/nx)
            self.weights['b' + str(layer)] = np.zeros((0, 1))
