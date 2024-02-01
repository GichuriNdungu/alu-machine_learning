#!/usr/bin/env python3
'''function that conducts forward propagation using Dropout:'''
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    ''''args: X: numpy.ndarray of shape (nx, m) containing the input data for
    the network, nx: number of input features to the network, m: number of
    data points, weights: dictionary of the weights and biases of the neural
    network, L: number of layers in the network, keep_prob: probability that
    a node will be kept'''
    outputs = {}
    outputs['A0'] = X
    for layer in range(L):
        # define the weights and biases
        weight = weights['W{}'.format(layer+1)]
        bias = weights['b{}'.format(layer+1)]
        # calculate layer output as before
        linear_reg = np.dot(weight, outputs['A{}'.format(layer)])
        z = np.add(linear_reg, bias)
        # Get randomized 1 and 0's
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        #  check if layer is final
        if layer < L:
            # apply tanh activation
            A = np.tanh(z)
            # multiply the output of the layer to the dropout
            A *= dropout
            # scale A
            A /= keep_prob
            # outputs['D{}'.format(layer)] = dropout
        else:
            # apply softmax activation for the output layer
            A = np.exp(z)
            A /= np.sum(A, axis=0, keepdims=True)
        outputs['A{}'.format(layer+1)] = A
    return outputs
