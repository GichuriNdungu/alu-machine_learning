#!/usr/bin/env python3
''''function to calculate the
 cost after l2-regularization'''
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''args: cost: cost of the network before regularization
       lambtha: regularization parameter
       weights: dictionary of the weights and biases (numpy.ndarrays)
       L: number of layers in the neural network
       m: number of data points used
       returns: cost of the network accounting for L2 regularization'''
    for w in range(1, L + 1):
        w = "W" + str(w)
        cost += (lambtha / (2 * m)) * np.linalg.norm(weights[w])
    return cost
