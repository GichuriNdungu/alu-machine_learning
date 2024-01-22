#!/usr/bin/env python3'''
'''function that trains the model'''
import tensorflow as tf


def create_train_op(loss, alpha):
    '''args:loss:loss of the network’s prediction
            alpha:learning rate
            returns:an operation that trains the network using gradient descent'''
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
