#!/usr/bin/env python3
'''Function that creates a layer with l2 regularization'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''args: prev: output from previous layer,
    n: nodes in layer,
    activation: activation function for our layer,
    lambtha: regularization parameter'''
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    l2_loss = tf.contrib.layers.l2_regularizer(scale=lambtha)
    hidden_layer = tf.layers.Dense(units=n, activation=activation,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=l2_loss)
    output = hidden_layer(prev)
    return output
