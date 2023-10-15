#!/usr/bin/env python3
'''A function that calculates the minor of a matrix'''


def inverse(matrix):
    '''Invert a matrix based on the adjugate'''
    # divide 1 with the determinant
    # check for singularity
    if determinant(matrix) == 0:
        return None
    else:
        mat_div = (1/determinant(matrix))
        # multiply mat_div with adjugate
        inverse = []
        for element in adjugate(matrix):
            result = []
            for inner_element in element:
                result.append(mat_div * inner_element)
            inverse.append(result)
    return inverse


def adjugate(matrix):
    '''extract the minor of matrix'''
    if type(matrix) != list:
        raise TypeError('matrix must be a list of lists')
    n = len(matrix)
    if n == 0:
        raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if type(row) != list:
            raise TypeError('matrix must be a list of lists')
        if len(row) != n:
            raise ValueError('matrix must be a non-empty square matrix')
        if len(matrix[0]) == 0:
            raise ValueError('matrix must be a non-empty square matrix')
    if n == 1:
        return [[1]]
    # if n == 2:
    #     minor = []
    #     row_a = [matrix[1][1], matrix[1][0]]
    #     minor.append(row_a)
    #     row_b = [matrix[0][1], matrix[0][0]]
    #     minor.append(row_b)
    #     return minor
    #  modify else to handle 2*2 as well
    else:
        minor = []
        for row_i in range(n):
            minor_row = []
            for j in range(n):
                # determine sign based on position
                sign = (-1) ** (row_i + j)
                submatrix = []
                for row in range(n):
                    if row == row_i:
                        continue
                    new_row = []
                    for column in range(n):
                        if column == j:
                            continue
                        new_row.append(matrix[row][column])
                    submatrix.append(new_row)
                minor_row.append(sign * determinant(submatrix))
            minor.append(minor_row)
            mat_adjugate = list(map(list, zip(*minor)))
        return mat_adjugate


def determinant(matrix):
    '''calculates the determinant of a matris
    args: matrix
    return: determinant'''
    if not isinstance(matrix, list):
        raise TypeError('matrix must be a list of lists')
    n = len(matrix)
    if n == 0:
        raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) == 0 and n == 1:
            return 1
        if len(row) != n:
            raise ValueError('matrix must be a square matrix')
    if n == 1:
        return matrix[0][0]
    if n == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return (a*d) - (b*c)
    elif n == 3:
        a = matrix[0][0] * ((matrix[1][1] * (matrix[2][2])) -
                            ((matrix[1][2]) * (matrix[2][1])))
        b = matrix[0][1] * ((matrix[1][0] * (matrix[2][2])) -
                            ((matrix[1][2]) * (matrix[2][0])))
        c = matrix[0][2] * ((matrix[1][0] * (matrix[2][1])) -
                            ((matrix[1][1]) * (matrix[2][0])))
        determinant_value = a - b + c
        return determinant_value
    else:
        multiplier = 1
        d = 0
        for j in range(n):
            element = matrix[0][j]
            submatrix = []
            for row in range(n):
                if row == 0:
                    continue
                new_row = []
                for column in range(n):
                    if column == j:
                        continue
                    new_row.append(matrix[row][column])
                submatrix.append(new_row)
            d += (element * multiplier * determinant(submatrix))
            multiplier *= -1
        return d
