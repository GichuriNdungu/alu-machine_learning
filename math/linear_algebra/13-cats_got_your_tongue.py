#!/usr/bin/env python3
import numpy as np
"""
    Concatenates two matrices along a specific axis.

    Parameters:
        mat1 (numpy.ndarray):
        mat2 (numpy.ndarray):
        axis (int, optional):

    Returns:
        numpy.ndarray: The concatenated array.

    Example:
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[5, 6]])
        result = np_cat(mat1, mat2, axis=0)
        print(result)
    """


def np_cat(mat1, mat2, axis=0):
    '''args:
    mat 1, mat2, axis
    return: concatenated array'''
    return np.concatenate((mat1, mat2), axis)
