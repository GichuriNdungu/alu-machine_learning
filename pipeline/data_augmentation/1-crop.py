#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
def crop_image(image, size):
    '''args: image; image to be cropped
    size: crop_size'''
    seed = np.random.randint(1234)
    output = tf.random_crop(image, size=size)
    return output