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
        self.__W = np.random.randn(nx, 1)
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

    def forward_prop(self, X):
        '''A method to calculate the forward
        propagation of our neuron
        nx = number of features
        m = number of houses'''

        nx, m = X.shape
        '''first transpose the weights vector'''
        w_t = np.transpose(self.W)
        '''multiply transposed weights with input vector'''
        weighted_sum = np.dot(w_t, X)
        ''' Add the bias to the dot'''
        Z = np.add(weighted_sum, self.b)
        '''apply the activation function, sigmoid'''

        self.__A = 1/(1+np.exp(-Z))
        return self.__A

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
        '''Evaluates the predictions of the Neuron'''
        '''call the forward prop alg'''
        Y_pred = self.forward_prop(X)
        Y_convert = np.where(Y_pred >= 0.5, 1, 0)
        cost = self.cost(Y, Y_pred)
        return Y_convert, cost
