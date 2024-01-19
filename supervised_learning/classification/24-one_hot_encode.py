#!/usr/bin/env python3
import numpy as np
from tensorflow.keras.utils import to_categorical


def one_hot_encode(Y, classes):
    '''
    Converts a numeric label vector into a one-hot matrix.

    Args:
    - Y: numpy.ndarray with shape (m,) containing numeric class labels
    - classes: the maximum number of classes found in Y

    Returns:
    - a one-hot encoding of Y with shape (classes, m)
    - None on failure
    '''
    try:
        one_hot_matrix = to_categorical(Y, num_classes=classes).T
        return one_hot_matrix
    except Exception as e:
        print(f"Error: {e}")
        return None
