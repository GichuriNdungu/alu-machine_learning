#!/usr/bin/env python3
'''function to normalize an input x'''


def normalize(X, m, s):
    '''args: X; input
    m: mean of X
    s: standard dev of x 
    returns: Z_score(standardized X)'''
    for element in range(len(X)):
        z = (X[element-1]-m)/s
    return z
