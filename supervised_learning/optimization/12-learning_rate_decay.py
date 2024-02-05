#!/usr/bin/env python3
'''function that implements learning rate decay'''

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''args: alpha: initial learning rate
            decay_rate: weight to determin the rate of decay
            global_step: number of passes of gradient descent that have lapsed
            decay_step: number of passes of gradient descent before decay
            returns: updated alpha after each update'''
    return tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
