#!/usr/bin/env python
import tensorflow as tf
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    outputs = {}
    outputs['A0'] = X
    for layer in range(1, L+1):
        # define the weights and biases
        weight = weights['W{}'.format(layer)]
        bias = weights['b{}'.format(layer)]
        # calculate layer output as before
        linear_reg = np.dot(weight, outputs['A{}'.format(layer-1)])
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
        outputs['A{}'.format(layer)] = A
    return outputs
