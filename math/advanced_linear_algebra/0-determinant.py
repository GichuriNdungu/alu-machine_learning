#!/usr/bin/env python3
def determinant(matrix):
    '''check whether matrix is a list of lists'''
    if len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    if len(matrix[0]) == 0:
        return 1
    elif len(matrix[0]) == 1:
        return matrix[0][0]
    elif isinstance(matrix, list):
        '''check whether matrix is a square'''
        for row in matrix:
            if not isinstance(row, list):
                raise TypeError('matrix must be a list of lists')
        if len(matrix[0]) != len(matrix):
            raise ValueError('matrix must be a square matrix')
        else:
            if len(matrix) == 2:
                a = matrix[0][0] - matrix[1][1]
                b = matrix[0][1] - matrix[1][0]
                determinant =a -b
                return determinant
            else:
                a = matrix[0][0] * ((matrix[1][1]* (matrix[2][2])) - ((matrix[1][2]) * (matrix[2][1])))
                b = matrix[0][1] * ((matrix[1][0]* (matrix[2][2])) - ((matrix[1][2]) * (matrix[2][0])))
                c = matrix[0][2] * ((matrix[1][0]* (matrix[2][1])) - ((matrix[1][1]) * (matrix[2][0])))
                determinant = a - b +c
                return determinant