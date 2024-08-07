#!/usr/bin/env python3
'''Defines a Neural Network with one hidden
layer performing binary classification'''

import numpy as np
import matplotlib.pyplot as plt


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
        '''The weights vector for the hidden layer. '''
        self.__W1 = np.random.randn(nodes, nx)
        '''' The bias for the hidden layer'''
        self.__b1 = np.zeros((nodes, 1))
        '''The activated output for the hidden layer.'''
        self.__A1 = 0
        ''''The weights vector for the output neuron'''
        self.__W2 = np.random.randn(1, nodes)
        '''The bias for the output neuron.'''
        self.__b2 = 0
        ''''The activated output for the output neuron (prediction'''
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

    def cost(self, Y, A):
        '''cost function for our Neuron'''

        m = A.shape[1]
        '''use binary cross Entropy function to calculate logloss'''
        # # Add epsilon to avoid taking log of zero
        # epsilon = 1e-15
        # A_clipped = np.clip(A, epsilon, 1 - epsilon)
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        total_loss = np.sum(loss)

        '''divive the loss by the num of examples'''

        cost_func = total_loss/m

        return cost_func

    def evaluate(self, X, Y):
        '''evaluates the predictions of the neural net'''
        # first perform forward propagation
        _, Y_pred = self.forward_prop(X)
        # then calculate the cost of running the net
        cost = self.cost(Y, Y_pred)
        # change the values
        y_convert = np.where(Y_pred >= 0.5, 1, 0)
        return y_convert, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        ''''calculates the gradient descent
          of the neural network and updates
          values of the weights and biases in a single pass'''
        # first get the cost at the output layer
        m = X.shape[1]
        dz2 = A2 - Y
        dw2 = np.dot(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        # then get the cost at the hidden layer
        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        # Adjust the weights and the biases using logistic regression
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        '''trains the neural net'''
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        cost_after_iteration =[]
        for iteration in range(iterations):
            # do forward prop and get the output after first
            A1, A2 = self.forward_prop(X)
            # calculate the gradient descent
            self.gradient_descent(X, Y, A1, A2, alpha)
            self.__A1 = A1
            self.__A2 = A2
            cost = self.cost(Y,A2)
            if iteration%step == 0:
                cost_after_iteration.append(cost)
            if verbose == True and iteration%step == 0:
                if type(step) != int:
                    raise TypeError('Step must be an integer')
                if step <0 or step > iterations:
                    raise ValueError('step must be positive and <= iterations')
                print(f'cost after {iteration} iterations: {cost}')  
        #graph the results
        if graph == True and cost_after_iteration: 
            ''' print a graph of the cost after x iterations'''
            if type(step) != int:
                raise TypeError('step must be an integer')
            if step <0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
            x = np.arange(0, iterations, step)
            y = cost_after_iteration
            plt.plot(x, cost_after_iteration, color='blue')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training cost')
        # return evaluation
        return self.evaluate(X, Y)

