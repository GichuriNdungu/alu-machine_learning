#!/usr/bin/env python3
'''function that implements learning rate decay'''
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''args: alpha: initial learning rate
            decay_rate: weight to determin the rate of decay
            global_step: number of passes of gradient descent that have lapsed 
            decay_step: number of passes of gradient descent before decay
            returns: updated alpha after each update'''
    alpha /= 1+(decay_rate*np.floor(global_step/decay_step))
    return alpha
