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

    Args:
        Y(numpy.ndarray): one-hot matrix with the correct labels
        weights(dict): The weights and biases of the network
        cache(dict): The outputs of each layer of the network
        alpha(float): The learning rate
        lambtha(float): The L2 regularization parameter
        L(int): The number of layers of the network
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dw = (1 / m) * np.matmul(dz, A.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(W.T, dz) * (A * (1 - A))
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dw
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db

    return weights
