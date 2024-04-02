#!/usr/bin/env python3

'''defines a function that shears an image 
based on a certain level of intensity'''

import tensorflow as tf
def shear_image(image, intensity):
    '''args: image; image to shear
    intensity; intensity for which the image will be sheared'''
    output = tf.keras.preprocessing.image.random_shear(image, intensity=intensity)
    return output
