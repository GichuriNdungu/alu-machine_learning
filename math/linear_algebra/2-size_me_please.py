#!/usr/bin/env python3
'''
Matrix Shape Determination Function

This script defines a function for determining the shape (dimensions) of a matrix. The function can determine the shape of a matrix whether it's a 1D, 2D, or 3D matrix.

Usage:
    shape = matrix_shape(matrix)

Parameters:
    matrix (list): The input matrix represented as a list of lists. It can be a 1D, 2D, or 3D matrix.

Returns:
    shape (list): A list representing the shape of the matrix:
        - For a 1D matrix, the shape is [n], where n is the number of elements.
        - For a 2D matrix, the shape is [m, n], where m is the number of rows and n is the number of columns.
        - For a 3D matrix, the shape is [m, n, p], where m is the number of layers, n is the number of rows in each layer, and p is the number of columns in each layer.

Notes:
    - The function uses type checking and isinstance to determine the matrix's structure.
    - It can handle matrices consisting of integers and floating-point numbers.

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
