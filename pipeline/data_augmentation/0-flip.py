#!/usr/bin/env python3
'''defines a function that flips an image horizontally'''
import tensorflow as tf
def flip_image(image):
    '''args: image
    return: horizontally flipped image'''
    flipped = tf.image.flip_left_right(image)
    return flipped