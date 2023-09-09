#!/usr/bin/env python3
'''function that concatenates two matrices
along the same axis'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''contanate matrices on a given axis'''
    if len(mat1) == 0 or len(mat2) == 0:
        return None
    if axis == 0:
        result = []
        for row in mat1:
            result.append(row)
        for row in mat2:
            result.append(row)
        return result
    elif axis == 1:
        result = []
        for row1, row3 in zip(mat1, mat2):
            new_row = row1+row3
            result.append(new_row)
        return result
    else:
        return None


mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat4)
print(mat5)
mat1[0] = [9, 10]
mat1[1].append(5)
print(mat1)
print(mat4)
print(mat5)
