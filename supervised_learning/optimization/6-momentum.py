#!/usr/bin/env python3
'''function that creates a training operation for
 neural net in tensorflow
 using grad descent with momentum optimization'''
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    '''args:loss;loss of the net
            alpha: learning rate
            beta1: momentum weight
            return:momentum optimization operation'''
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train_op = optimizer.minimize(loss)
    return train_op
