#!/usr/bin/env python3'''
'''function that calculates the accuracy of a prediction'''
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    '''args:y:placeholder for the labels of the input data
            y_pred:tensor containing the network’s predictions
            returns:tensor containing the decimal accuracy of the prediction'''
    value_accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(value_accuracy, tf.float32))
    return accuracy
