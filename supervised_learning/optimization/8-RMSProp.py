#!/usr/bin/env python3
'''function that creates a training operation for
 neural net in tensorflow
 using grad descent with rmsprop optimization'''
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''args:loss;loss of the net
            alpha: learning rate
            beta2: rmsprop weight
            return:rmsprop optimization operation'''
    optimizer = tf.train.RMSPropOptimizer(
        alpha, beta2, epsilon=epsilon).minimize(loss)
    return optimizer
