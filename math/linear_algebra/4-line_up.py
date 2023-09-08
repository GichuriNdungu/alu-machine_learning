#!/usr/bin/env python3
'''
Function def add_arrays(arr1, arr2):
adds two arrays element-wise:'''


def add_arrays(arr1, arr2):
    # Check if the arrays have the same shape
    if len(arr1) != len(arr2):
        return None

    # Initialize a result list to store the element-wise sum
    result = []

    # Perform element-wise addition and store the result
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result


arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
print(add_arrays(arr1, arr2))
print(arr1)
print(arr2)
print(add_arrays(arr1, [1, 2, 3]))
