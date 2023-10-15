#!/usr/bin/env python3

"""
modeule with function definiteness(matrix)
"""

import numpy as np


def definiteness(matrix):
    """
    calculate the definiteness of a matrix
    """

    n, p = np.linalg.eig(matrix)  # n is eigenvalues, p is eigenvectors
    # if matrix is not a numpy.ndarray raise type error
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if not matrix.any():
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.shape[0] == 1:
        if n[0] > 0:
            return "Positive definite"
        elif n[0] == 0:
            return "Positive semi-definite"
        else:
            return "Negative definite"
    if matrix.shape[0] == 2:
        if n[0] > 0 and n[1] > 0:
            return "Positive definite"
        elif n[0] < 0 and n[1] < 0:
            return "Negative definite"
        elif n[0] == 0 and n[1] == 0:
            return "Positive semi-definite"
        else:
            return "Indefinite"
    if matrix.shape[0] == 3:
        if n[0] > 0 and n[1] > 0 and n[2] > 0:
            return "Positive definite"
        elif n[0] < 0 and n[1] < 0 and n[2] < 0:
            return "Negative definite"
        elif n[0] == 0 and n[1] == 0 and n[2] == 0:
            return "Positive semi-definite"
        else:
            return "Indefinite"

    return None
