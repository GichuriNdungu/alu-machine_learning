#!/usr/bin/env python3
#import numpy as np
def matrix_shape(matrix):
    shape = []
    if type(matrix[0]) == list and len(matrix[0]) == 2:
        shape.append(len(matrix))
        shape.append(len(matrix[0]))
    elif type(matrix[0]) == list:
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
#def matrix_shape (matrix):
    shape = list(np.shape(matrix))
  
    return shape