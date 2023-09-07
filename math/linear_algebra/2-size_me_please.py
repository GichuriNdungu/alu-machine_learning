#!/usr/bin/env python3
'''
Matrix Shape Determination Function

This script defines a function for determining the shape (dimensions) of a matrix.
The function can determine the shape of a matrix whether it's a 1D, 2D, or 3D matrix.

Usage:
    shape = matrix_shape(matrix)

Parameters:
    matrix (list): The input matrix represented as a list of lists. It can be a 1D, 2D, or 3D matrix.
Example:
    matrix = [[1, 2, 3], [4, 5, 6]]
    shape = matrix_shape(matrix)
    print("Shape of the matrix:", shape)  # Output: [2, 3]
'''


def matrix_shape(matrix):
    '''determine the shape of a matrix'''
    shape = []
    # check if matrix is 1d
    if all(isinstance(obj, (int, float)) for obj in matrix):
        shape.append(len(matrix))
    # check if matrix is 2d
    elif type(matrix[0]) == list and type(matrix[0][0]) == int:
        shape.append(len(matrix))
        shape.append(len(matrix[0]))
    # check if matrix is higher dimension
    else:
        if not isinstance(matrix[0][0], list):
            return []
        else:
            current_dimension_size = len(matrix)
            '''use recursion to determine the shape of the rem dimensions'''
            rem_dimensions_shape = matrix_shape(matrix[0])

            shape = [current_dimension_size,] + rem_dimensions_shape
    return shape
