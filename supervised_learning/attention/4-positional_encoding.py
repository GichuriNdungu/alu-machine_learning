#!/usr/bin/env python3

import numpy as np


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
            PE[pos, i] = np.sin(pos / (10000 ** (i/dm)))
            if i+1 < dm:
                PE[pos, i+1] = np.cos(pos / (10000**((i+1)/dm)))
    return PE
