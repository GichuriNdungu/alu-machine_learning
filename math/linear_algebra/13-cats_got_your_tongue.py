#!/usr/bin/env python3
'''This function concatenates two matrices on a given axis
params: matrix 1, matrix 2, axis
returns the concatenated array
'''
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''functiont to concatenate 2 arrays:
    returns an nd array of concatenation'''
    return np.concatenate((mat1, mat2), axis)
