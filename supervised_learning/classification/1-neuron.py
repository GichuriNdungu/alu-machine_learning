#!/usr/bin/env python3
'''Defines a binary classification class'''
import numpy as np


class Neuron:
    '''class neuron representing a single neuron in an NN'''

    def __init__(self, nx):
        '''nx refers to the number of input features'''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        '''declare private instance attributes'''
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''getter method for self.__w'''
        return self.__W

    @property
    def b(self):
        '''getter method for private instace b'''
        return self.__b

    @property
    def A(self):
        '''getter method for private instance A'''
        return self.__A
