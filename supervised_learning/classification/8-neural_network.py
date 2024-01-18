#!/usr/bin/env python3
'''Defines a Neural Network with one hidden layer performing binary classification'''

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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros(nodes, 1)
        self.A1 = 0
        self.w2 = np.random(1, nodes)
        self.b2 = 0
        self.A2 = 0
