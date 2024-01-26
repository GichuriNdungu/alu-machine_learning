#!/usr/bin/env python3
'''function that returns the tensor of the output layer'''
import tensorflow as tf


def create_layer(prev, n, activation):
    '''args:prev:tensor output of the previous layer
            n:number of nodes in the layer to create
            activation:activation function that the layer should use
            returns:the tensor of the output layer'''
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name='layer')
    output = layer(prev)
    return output
