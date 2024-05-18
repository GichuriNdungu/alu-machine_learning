#!/usr/bin/env python3
'''class NST that performs neural style transfer'''
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
        self.load_model()
        self.gram_style_features = []
        self.content_feature = None
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
                new_height = int(original_height * (512 / original_width))
            size = (new_height, new_width)
            resized_image = tf.image.resize_bicubic(np.expand_dims(image,
                                                                   axis=0),
                                                    size)

            # clip the pixel values to [0, 1]
            resized_image = resized_image / 255
            resized_image = tf.clip_by_value(resized_image, 0.0, 1.0)

            # confirm that the new shape is (1, hnew, w_new, 3)
            resized_image = tf.ensure_shape(resized_image, [1, None, None, 3])

            return resized_image

    def load_model(self):
        '''loads a VGG19 model for neural transfer'''
        # define the base_model
        VGG19_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        # save model
        VGG19_model.save('VGG19_base_model')
        # Add customizable objects to model
        # Here we are replacing any maxpooling layer
        # in our model with average pooling
        # load the model afresh with the customs
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        loaded_model = tf.keras.models.load_model("VGG19_base_model.h5",
                                                  custom_objects)

        # define a list for outputs:
        style_output = []
        content_output = None
        # check whether layer name is self.style_layers
        for layer in loaded_model.layers:
            if layer.name in self.style_layers:
                style_output.append(layer.output)
            if layer.name in self.content_layer:
                content_output = (layer.output)
            layer.trainable = False
        output = style_output + [content_output]
        model = tf.keras.models.Model(loaded_model.input, output)
        self.model = model
        return self.model

    @staticmethod
    def gram_matrix(input_layer):
        '''function to calculate the gram matrix of input'''
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or\
                tf.rank(input_layer)._numpy() != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # get input dimensions
        _, h, w, c = input_layer.shape
        # flatten the dimensions h and w
        pd = (h * w)
        features = tf.reshape(input_layer, (pd, c))
        # calculate gram matrix (transpose of flat)
        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram /= tf.cast(pd, tf.float32)

        return gram
    def generate_features(self):
        '''function to extract style and content features'''
        model = self.load_model()
        for layer in self.model.layers:
            if layer.name in self.style_layers:
                output = layer.output
                output_gram = self.gram_matrix(output)
                self.gram_style_features.append(output_gram)
            if layer.name in self.content_layer:
                output = layer.output
                self.content_feature = output
            

