#!/usr/bin/env python3
'''function to normalize an input x'''


def normalize(X, m, s):
    '''args: X; input
    m: mean of X
    s: standard dev of x
    returns: Z_score(standardized X)'''
    z_score = (X-m)/s
    return z_score
