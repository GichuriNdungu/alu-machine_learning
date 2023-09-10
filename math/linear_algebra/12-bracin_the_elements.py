#!/usr/bin/env python3
import numpy as np
'''perform additions, subtractions 
multiplications and divisions on arrayys'''


def np_elementwise(mat1, mat2):
    '''args: matrix 1 matrix 2
    return: add:
            subtract:
            mutiple: 
            div:'''
    result = mat2 + mat1, mat1 - mat2, mat1*mat2, mat1/mat2
    return result
