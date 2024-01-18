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

    def forward_prop(self, X):
        '''A method to calculate the forward
        propagation of our neuron
        nx = number of features
        m = number of houses'''

        nx, m = X.shape
        '''multiply transposed weights with input vector'''
        weighted_sum = np.dot(self.W, X)
        ''' Add the bias to the dot'''
        Z = np.add(weighted_sum, self.b)
        '''apply the activation function, sigmoid'''

        self.__A = 1/(1+np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        '''cost function for our Neuron'''

        'get corresponding elements in actual and pred'
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''calculates one pass of gradient
        descent on the neuron'''
        m = X.shape[1]
        cost = A - Y
        dw = 1/m * (np.matmul(cost, X.T))
        db = np.sum(cost) / m

        '''update the weights and biases'''
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''Implement back propagation for full training'''
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for iteration in range(iterations):
            '''Do forward propagation'''
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        evaluation_results = self.evaluate(X, Y)
        return evaluation_results
