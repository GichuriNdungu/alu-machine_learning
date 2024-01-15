#!/usr/bin/env python3
'''Defines a binary classification class'''
import numpy as np


class Neuron:
    '''class neuron representing a single neuron in an NN'''

    def __init__(self, nx):
        '''nx refers to the number of input features'''
        if type(nx) != int:
            raise Exception('nx must be an integer')
        if nx < 1:
            raise Exception('nx must be a positive integer')
        '''declare a vector size'''
        vector_size = np.random.randint(1, 100)
        self.W = np.random.normal(size=vector_size)
        self.b = 0
        self.A = 0
