#!/usr/bin/env python3
'''function that creates a training operation for
 neural net in tensorflow
 using grad descent with adam optimization'''
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''args:loss;loss of the net
            alpha: learning rate
            beta1: momentum weight
            beta2: rmsprop weight
            return:rmsprop optimization operation'''
    optimizer = tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon=epsilon).minimize(loss)
    return optimizer
