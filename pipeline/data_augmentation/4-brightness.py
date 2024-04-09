#!/usr/bin/env python3

'''defines a function that increases
 the brightness of an image'''
import tensorflow as tf
def change_brightness(image, max_delta):
    '''args: image; input image
    max_delta: '''
    output = tf.image.adjust_brightness(image=image, delta=max_delta)
    return output
