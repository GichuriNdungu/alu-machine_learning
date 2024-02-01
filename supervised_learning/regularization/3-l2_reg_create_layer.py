#!/usr/bin/env python3
'''Function that creates a layer with l2 regularization'''
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''args: prev: output from previous layer,
    n: nodes in layer,
    activation: activation function for our layer,
    lambtha: regularization parameter'''
    hidden_layer = tf.layers.dense(input=prev, units=n, activation=activation,
                                   kernel_regularizer=tf.contrib.layers.l2_regularize(scale=lambtha))
    return hidden_layer
