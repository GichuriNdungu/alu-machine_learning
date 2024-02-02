#!/usr/bin/env python3
'''Function that creates a layer with l2 regularization'''
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''args: prev: output from previous layer,
    n: nodes in layer,
    activation: activation function for our layer,
    keep_prob: probability that a node will be kept'''
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    dropout = tf.layers.Dropout(keep_prob)
    hidden_layer = tf.layers.Dense(units=n, activation=activation,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=dropout)
    output = hidden_layer(prev)
    return output
