#!/usr/bin/env python3
'''Function that transposes a matrix'''


def matrix_transpose(matrix):
    """
    Transpose a 2D matrix.

    Args:
        matrix (list of lists)
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
