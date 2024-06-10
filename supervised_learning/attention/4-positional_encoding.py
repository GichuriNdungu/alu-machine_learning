#!/usr/bin/env python3

import numpy as np

def get_angle(pos, i, dm):
    '''calculates the angles for positional encoding formulas
    PE(pos, 2i) = sin(pos / 10000^(2i/dm))
    PE(pos, 2i +1) = cos(pos / 10000^(2i/dm))'''

    angle_rates = 1 /(10000 ** (i / dm))
    return pos * angle_rates

def positional_encoding(max_seq_len, dm):
    '''calculates the positional encoding for a transformer
    params:
        max_seq_len: maximum sequence length
        dm: model depth
    return:
        np array with shape(max_seq_len, dm)
            array is positional encoding vectors'''
    PE = np.zeros([max_seq_len, dm])
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            PE[pos, i] = np.sin(get_angle(pos, i, dm))
            PE[pos, i+1] = np.cos(get_angle(pos, i, dm))
    return PE
