#!/usr/bin/env python3
'''function to get the mean and
 std for future normalization'''
import numpy as np


def normalization_constants(X):
    '''args: X
    return: X.mean, X.std'''
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    return mean, std_dev
