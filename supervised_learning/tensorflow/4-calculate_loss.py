#!/usr/bin/env python3'''
'''function that calculates the loss of a prediction'''
import tensorflow as tf


def calculate_loss(y, y_pred):
    '''args:y:placeholder for the labels of the input data
            y_pred:tensor containing the network’s predictions
            returns:tensor containing the loss of the prediction'''
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=y_pred))
    return loss
