#!/usr/bin/env python3
'''function to get the summation of a squares
\sum_{i=1}^{n} i^2:'''


def summation_i_squared(n):
    '''function to get the summation of a squares
\sum_{i=1}^{n} i^2:'''
    if type(n) != int:
        return None
    else:
        squared_num = n*((n+1)*(2*n+1))//6
        return squared_num
