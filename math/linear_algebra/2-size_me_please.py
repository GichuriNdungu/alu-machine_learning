#!/usr/bin/env python3

def matrix_shape(matrix):
    shape = []
    if all(isinstance(obj, (int, float)) for obj in matrix):
        shape.append(len(matrix))
    elif type(matrix[0]) == list and type(matrix[0][0]) == int:
        shape.append(len(matrix))
        shape.append(len(matrix[0]))
    elif type(matrix[0][0]) == list:
        num_rows = len(matrix)
        num_of_columns = len(matrix[0])
        spec_elements = len(matrix[0][0])
        shape.append(num_rows)
        shape.append(num_of_columns)
        shape.append(spec_elements)
    elif type(matrix) == list:
        shape.append(len(matrix))
    else:
        shape.append(num_rows)
        shape.append(num_of_columns)
    return shape
