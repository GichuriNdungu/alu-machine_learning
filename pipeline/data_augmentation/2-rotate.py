#!/usr/bin/env python3

'''defines a function that rotates an
 image by 90 degrees counter-clockwise'''

import tensorflow as tf
def rotate_image(image):
    '''args: image: image to be cropped'''
    rot_90 = tf.image.rot90(image, k=1)