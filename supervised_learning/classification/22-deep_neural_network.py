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
            r = layers[layer]
            c = layers[layer-1]
            self.weights[
                'W' + str(layer+1)] = np.random.randn(r,
                                                      c) * np.sqrt(
                                                          2/layers[layer-1])
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

    def forward_prop(self, X):
        '''calculates the forward propagation for the neural net'''
        m = list(X.shape)
        if m[0] != 0:
            self.__cache['A0'] = X
        for layer in range(1, self.__L + 1):
            # Linear transformation (Z = W*X + b)
            W = self.__weights['W'+str(layer)]
            A = self.__cache['A' + str(layer-1)]
            b = self.__weights['b' + str(layer)]
            weighted_sum = np.dot(W, A)
            Z = np.add(weighted_sum, b)

            # Activation function
            A = 1 / (1 + np.exp(-Z))

            # Save the intermediate values in the cache
            self.__cache['A' + str(layer)] = A
        output = self.__cache['A' + str(self.__L)]
        return output, self.cache

    def cost(self, Y, A):
        ''''calculates the cost of the model'''
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost

    def evaluate(self, X, Y):
        ''''evaluates the model'''
        # first perform a forward propagation
        self.forward_prop(X)
        # then get the output from the cache
        model_out = self.__cache['A' + str(self.__L)]
        cost = self.cost(Y, model_out)
        y_convert = np.where(model_out >= 0.5, 1, 0)
        return y_convert, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''calculates the gradient descent of neural net after 1 pass'''
        m = Y.shape[1]
        # Backward propagation
        dZ_last = cache['A' + str(self.__L)] - Y
        for layer in range(self.__L, 0, -1):
            A_prev = cache['A' + str(layer-1)]
            dW = (1 / m) * np.dot(dZ_last, A_prev.T)
            db = (1 / m) * np.sum(dZ_last, axis=1, keepdims=True)
            dZ_prev = np.dot(
                self.__weights['W' + str(
                    layer)].T, dZ_last) * (A_prev * (1 - A_prev))

            # Update weights and biases
            self.__weights['W' + str(layer)] -= alpha * dW
            self.__weights['b' + str(layer)] -= alpha * db

            dZ_last = dZ_prev

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''Trains the deep neural network'''
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for iteration in range(iterations):
            # do forward prop and get the output after first
            self.forward_prop(X)
            # calculate the gradient descent
            self.gradient_descent(Y, self.__cache, alpha)
        # return evaluation
        return self.evaluate(X, Y)
