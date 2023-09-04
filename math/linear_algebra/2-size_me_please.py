#!/usr/bin/env python3
import numpy as np
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
def matrix_nam(matrix):
    shape = list(np.shape(matrix))
  
    return shape

matrix =  [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_nam(matrix))