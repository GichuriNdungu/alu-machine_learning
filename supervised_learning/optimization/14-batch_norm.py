#!/usr/bin/env python3
'''function that creates a layer using
 batch normalization with tensorflow'''
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    '''args: prev: is the activated output of the previous layer
            n: number of nodes in the layer to be created
            activation: activation function that
              should be used on the output of the layer
            returns:tensor of the activated output of the layer'''
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None,
                        kernel_initializer=initializer)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    offset = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    epsilon = 1e-8
    normalization = tf.nn.batch_normalization(
        x_prev, mean, variance, offset, scale, epsilon)
    return activation(normalization)
