#!/usr/bin/env python3
'''function that shuffles two arrays'''
import numpy as np


def shuffle_data(X, Y):
    '''args: X, y
    return: x_shuffle, y_shuffle'''
    zipped = list(zip(X, Y))
    np.random.shuffle(zipped)
    x_shuffle, y_shuffle = zip(*zipped)
    x_shuffle = np.array(x_shuffle)
    y_shuffle = np.array(y_shuffle)
    return x_shuffle, y_shuffle
