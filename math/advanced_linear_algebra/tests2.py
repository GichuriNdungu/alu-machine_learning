#!/usr/bin/env python3
matrix = [[5, 7, 9], 
        [3, 1, 8], 
        [6, 2, 4]]
a = matrix[0][0] * ((matrix[1][1]* (matrix[2][2])) - ((matrix[1][2]) * (matrix[2][1])))
b = matrix[0][1] * ((matrix[1][0]* (matrix[2][2])) - ((matrix[1][2]) * (matrix[2][0])))
c = matrix[0][2] * ((matrix[1][0]* (matrix[2][1])) - ((matrix[1][1]) * (matrix[2][0])))
print(c)