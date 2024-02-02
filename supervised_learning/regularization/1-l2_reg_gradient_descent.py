#!/usr/bin/env python3
"""
Defines a function that updates the weights and biases
using gradient descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization

    parameters:
        Y [one-hot numpy.ndarray of shape (classes, m)]:
            contains the correct labels for the data
            classes: number of classes
            m: number of data points
        weights [dict]: dictionary of weights and biases for the network
        cache [dict]: dictionary of the outputs of each layer of the network
        alpha [float]: learning rate
        lambtha: the regularization parameter
        L: the number of layers in the neural network

    Neural network using tanh activations on each layer except the last.
    Last layer uses softmax activation.
    """
    m = Y.shape[1]
    # get the derivative of the cost function with respect to output of last layer
    dz_last = cache['A{}'.format(L)] - Y
    # loop through the layers in reverse order
    for layer in range(L):
        if layer != L-1:
            # calculate the derivative of the output of the current layer
            # with respect to the output of the previous layer
            dz_prev = np.dot(weights['W{}'.format(layer + 1)].T, dz_last) * (
                1 - (cache['A{}'.format(layer + 1)] ** 2))
            # calculate the L2 regularization term
            l2 = (lambtha / m) * weights['W{}'.format(layer + 1)]
            # calculate the derivative of the weights
            dw = (1 / m) * np.dot(dz_last, cache['A{}'.format(layer)].T) + l
            # calculate the derivative of the biases
            db = (1 / m) * np.sum(dz_last, axis=1, keepdims=True)
            # update the weights and biases
            weights['W{}'.format(layer + 1)] = weights['W{}'.format(layer + 1)] - alpha * dw
            weights['b{}'.format(layer + 1)] = weights['b{}'.format(layer + 1)] - alpha * db
            # set the derivative of the output of the current layer
            # with respect to the output of the previous layer
            dz_last = dz_prev
        else:
            # calculate the derivative of the weights
            dw = (1 / m) * np.dot(dz_last, cache['A{}'.format(layer - 1)].T) + l2
            # calculate the derivative of the biases
            db = (1 / m) * np.sum(dz_last, axis=1, keepdims=True)
            # update the weights and biases
            weights['W{}'.format(layer + 1)] = weights['W{}'.format(layer + 1)] - alpha * dw
            weights['b{}'.format(layer + 1)] = weights['b{}'.format(layer + 1)] - alpha * db
