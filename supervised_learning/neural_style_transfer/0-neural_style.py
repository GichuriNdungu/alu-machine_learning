ss NST that performs neural style transfer'''
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
        if not isinstance(content_image, np.ndarray) or\
                content_image.ndim != 3 or\
                content_image.shape[-1] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        '''Rescale an image's pixels to 0 and 1.
        largest_image_size = 512 px
        args: image (image to rescale)
        return: rescaled image'''
        if not isinstance(image, np.ndarray) or\
                image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')
        else:
            # get the initial dimensions
            original_height, original_width, dim = image.shape
            # calculate the new dimensions

            if original_height > original_width:
                new_height = 512
                new_width = int(original_width * (512/original_height))
            else:
                new_width = 512
                new_height = int(original_height * (512 / original_height))

            resized_image = tf.image.resize_bicubic(np.expand_dims(image, axis=0),
                                                    size=(new_height, new_width))
            # resized_image = tf.image.convert_image_dtype(resized_image, tf.float32)
            # clip the pixel values to [0, 1]
            resized_image = resized_image / 255
            resized_image = tf.clip_by_value(resized_image, 0.0, 1.0)

            # confirm that the new shape is (1, hnew, w_new, 3)
            resized_image = tf.ensure_shape(resized_image, [1, None, None, 3])

            return resized_image

