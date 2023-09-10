#!/usr/bin/env python3
import numpy as np
'''function to concatenate 2 matrices
on a specific axis'''


def np_cat(mat1, mat2, axis=0):
    '''args:
    mat 1, mat2, axis
    return: concatenated array'''
    return np.concatenate((mat1, mat2), axis)
