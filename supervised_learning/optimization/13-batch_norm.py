#!/usr/bin/env python3
'''function that normalizes an unactivated output of
a neural network using batch normalization'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''args: z; array (m, n) that should be normalized
            gamma: aray(1, n) containing normalization scales
            beta: array(1, n) containing the offsets for normalization
            epsilon: factor to prevent 0-division error
            returns: normalized z matrix'''
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    # prevent 0-division error using epsilon
    std_dev = np.sqrt(var + epsilon)
    num = Z-mean
    std_data = num/std_dev
    normalized = gamma * std_data + beta
    return normalized
