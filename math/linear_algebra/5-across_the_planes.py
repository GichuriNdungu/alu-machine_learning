#!/usr/bin/env python3
'''
Function def add_arrays(arr1, arr2):
adds two arrays element-wise:'''


def add_matrices2D(mat1, mat2):
    '''function to add 2 arrays
    employs map and zip for zipping and summation'''
    if len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)
    return result
