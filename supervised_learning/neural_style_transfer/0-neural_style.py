#!/usr/bin/env python3
'''Class NST that performs neural style transfer'''
import numpy as np
import tensorflow as tf


class NST:
    '''class that performs neural style transfer'''

    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        '''class initializer'''
        if not isinstance(style_image, np.ndarray) or\
            style_image.ndim != 3 or\
            style_image.shape[-1] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        elif not isinstance(content_image, np.ndarray) or\
              content_image.ndim != 3 or\
                  content_image.shape[-1] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        elif not isinstance(alpha, int) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        elif not isinstance(beta, int) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        else:
            self.style_image = style_image
            self.content_image = content_image
            self.alpha = alpha
            self.beta = beta
        tf.enable_eager_execution()

    def scale_image(image):
        '''Rescale an image's pixels to 0 and 1.
        largest_image_size = 512 px
        args: image (image to rescale)
        return: rescaled image'''
        if not isinstance(image, np.ndarray) or\
              image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        else:
            # convert the image pixel values to requsite range (float 32- o and 1)
            image = tf.image.convert_image_dtype(image, tf.float32)
            # get the initial dimensions
            original_height, original_width = tf.shape(
                image)[0], tf.shape(image)[1]
            # calculate the new dimensions
            max_dim = 512
            scale = max_dim / tf.maximum(original_height, original_width)
            # we use cast since the values had originally been converted to floats
            new_height = tf.cast(original_height * scale, tf.float32)
            # both height and width will be the new dimensions that are int 32
            new_width = tf.cast(original_width * scale, tf.float32)

            # Use bicubic interpolation to resize our image according to the new dimensions
            resized_image = tf.image.resize(
                image, [new_height, new_width],
                  method=tf.image.ResizeMethod.BICUBIC)
            # add an extra batch dimension to allow for different neuralnet architectures
            tf.expand_dims(resized_image, axis=0)
            # confirm that the new shape is (1, hnew, w_new, 3)
            resized_image = tf.ensure_shape(resized_image, [1, None, None, 3])

            return resized_image
