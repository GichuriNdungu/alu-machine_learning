#!/usr/bin/env python3
mat4 = [[5, 7, 9, 3, 4], 
          [3, 1, 8, 4, 5], 
          [6, 2, 4, 5, 8],
          [3, 1, 8, 4, 5],
          [5, 7, 9, 3, 4] ]
n = len(matrix)
multiplier = 1
d = 0
for j in range(n):
    print(f'this is j {j}')
    element = matrix[0][j]
    print(f'this is the element {element}')
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
        print(f'this is the submatrix {submatrix}')
        # d += (element * determinant(submatrix))
        # multiplier *= -1
        # return (d)