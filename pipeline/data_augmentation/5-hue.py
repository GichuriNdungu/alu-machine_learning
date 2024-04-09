#!/usr/bin/env python3

'''defines a function that changes the hue of an image'''
import tensorflow as tf
def change_hue(image, delta):
    '''args: image; tensor containing image to change
    delta: amount of hue to change'''
    output = tf.image.adjust_hue(
    image, delta=delta)
    return output
