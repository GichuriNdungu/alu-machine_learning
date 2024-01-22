#!/usr/bin/env python3
''''function that returns two placeholders for our DNN'''
import tensorflow as tf


def create_placeholders(nx, classes):
    '''function that returns two placeholders,
      x and y, for the neural network

      args: nx: the number of feature columns in our data
            classes: the number of classes in our classifier
            returns: placeholders named x and y, respectively'''
    x = tf.placeholder('float', [None, nx], name='x')
    y = tf.placeholder('float', classes, name='y')
    return x, y
