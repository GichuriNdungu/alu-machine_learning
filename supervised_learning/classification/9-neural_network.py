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
            return self.__A1
