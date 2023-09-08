#!/usr/bin/env python3
def transpose_matrix(matrix):
    """
    Transpose a 2D matrix.

    Args:
        matrix (list of lists): The input matrix represented as a list of lists.

    Returns:
        list of lists: The transposed matrix.

    Example:
        matrix = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        transposed = transpose_matrix(matrix)
        # Output: [[1, 4], [2, 5], [3, 6]]
    """
    # Use the zip function to transpose the matrix
    transposed = list(map(list, zip(*matrix)))
    return transposed


matrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
print(transpose_matrix(matrix))

nested_list = [[1, 2], [3, 4], [5, 6]]

# Unpack elements from the nested list
unpacked_elements = [*nested_list[0], *nested_list[1], *nested_list[2]]

print(unpacked_elements)
