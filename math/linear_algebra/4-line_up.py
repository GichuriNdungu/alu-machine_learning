#!/usr/bin/env python3
'''
Function def add_arrays(arr1, arr2):
adds two arrays element-wise:'''


def add_arrays(arr1, arr2):
    '''function employs; 
    map func to parse sum to each of the elements
    and zip func to select the elements in their order'''
    if len(arr1) != len(arr2):
        return None
    else:
        new_list = list(map(sum, zip(arr1, arr2)))
        return new_list
