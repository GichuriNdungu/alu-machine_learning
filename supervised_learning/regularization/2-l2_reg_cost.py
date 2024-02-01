#!/usr/bin/env python3
'''Function to calculate the cost of a neural network with L2 regularization'''

import tensorflow as tf


def l2_reg_cost(cost):
    '''calculates the cost of a neural network with L2 regularization
    args: cost is a tensor containing the cost of the network without L2
    regularization
    returns: a tensor containing the cost of the network accounting for L2'''
    reg_loss = tf.losses.get_regularization_losses()
    total_loss = cost + reg_loss
    return total_loss
