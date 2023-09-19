#!/usr/bin/env python3
'''function to get the summation of a squares
'''


def summation_i_squared(n):
    '''function to get the summation of a squares
'''
    if n<1 or type(n) != int:
        return None
    else:
        squared_num = n*((n+1)*(2*n+1))//6
        return squared_num
