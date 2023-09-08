arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
matrices = [arr1, arr2]
for matrix in matrices:
    def matrix_shape(matrix):
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

                shape = [current_dimension_size, ] + rem_dimensions_shape
            print(shape)
matrix_shape()
