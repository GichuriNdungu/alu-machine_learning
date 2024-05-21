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
        self.generate_features()

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
        """
        Extracts the features used to calculate neural style cost

        Sets public instance attribute:
            gram_style_features and content_feature
        """
        VGG19_model = tf.keras.applications.vgg19
        preprocess_style = VGG19_model.preprocess_input(
            self.style_image * 255)
        preprocess_content = VGG19_model.preprocess_input(
            self.content_image * 255)

        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        gram_style_features = []
        for feature in style_features:
            gram_style_features.append(self.gram_matrix(feature))

        self.gram_style_features = gram_style_features
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        '''calculate the style cost of a single layer'''
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or\
                len(style_output.shape) is not 4:
            raise TypeError("style_output must be a tensor of rank 4")
        _, h, w, c = style_output.shape
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or\
                len(gram_target.shape) is not 3 or\
                gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]"
                .format(c, c))
        s = self.gram_matrix(style_output)
        g = gram_target

        # style cost = mean squared error of the difference between s and

        cost = tf.reduce_mean(tf.square(s-g))
        return cost

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image

        parameters:
            style_outputs [list of tf.Tensors]:
                contains stye outputs for the generated image

        returns:
            the style cost
        """
        length = len(self.style_layers)
        if type(style_outputs) is not list or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length))
        weight = 1 / length
        style_cost = 0
        for i in range(length):
            style_cost += (
                self.layer_style_cost(style_outputs[i],
                                      self.gram_style_features[i]) * weight)
        return style_cost

    def content_cost(self, content_output):
        '''calculates content cost for the generated image

        parameters:
            content_output Tensor with content output for the generated image
        returns:
            generated image content cost'''
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or content_output.shape != self.content_feature.shape:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))
        content_cost = tf.reduce_mean(
            tf.square(self.content_feature - content_output))
        return content_cost

    def total_cost(self, generated_image):
        '''calculates the total cost of the generated image

        parameters:
            generated_image: tensor containing the generated image
        return:
            J: total cost
            j_content: content_cost
            j_style: style cost'''

        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        # preprocess the generated image
        VGG19_model = tf.keras.applications.vgg19
        preprocess_generated = VGG19_model.preprocess_input(
            generated_image*255)

        # get the outputs using the created model
        outputs = self.model(preprocess_generated)
        style_outputs = outputs[:-1]
        content_outputs = outputs[-1]

        # get the total style cost for generated image
        j_style = self.style_cost(style_outputs)
        # get the total content cost for generate image
        j_content = self.content_cost(content_outputs)
        # add these two to get total cost
        j = j_style + j_content
        # return total cost
        return (j, j_content, j_style)
    def compute_grads(self, generated_image):
        '''computes gradients for the generated image
        parameters: 
            generated image
        return: 
            gradients: tf.Tensor containing the gradients for generated image
            j_total: total cost of the generated image 
            j_content: content cost for the generated image
            j_style: style cost for the generad image
        '''
        #check whether gen image is right shape and type
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image must be a tensor of shape {}".format(s))
        #call total_cost function
        
        # calculate the gradients using tf.GradientTape
        with tf.GradientTape() as tape:
            j_total, j_content, j_style = self.total_cost(generated_image)
        gradients = tape.gradient(j_total, generated_image)
        return gradients, j_total, j_content, j_style
    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        '''generates the neural style transfer image
        parameters: 
            iterations:number of iterations to perform gradient descent over
            step:step at which to print information about training process
            lr: learning rate for the gradient descent
            beta1: beta 1 parameter for the gradient descent
            beta2: beta 2 parameter for the gradient descent
        returns:
            generated_image: best generated image
            cost: best cost
            '''
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be positive")
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and (step < 0 or step > iterations):
            raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr < 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not (0<= beta1 <=1):
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not (0<=beta2<=1):
            raise ValueError("beta2 must be in the range [0, 1]")
        
        #initialize the best cost and the best image to keep track
        best_cost = float("inf")
        best_image = None

        generated_image = tf.Variable(self.content_image, dtype = tf.float32)

        #define the optimizer

        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
         # Define the training operation
        grads_placeholder = tf.placeholder(tf.float32, shape=generated_image.shape)
        apply_grads = optimizer.apply_gradients([(grads_placeholder, generated_image)])
        #initialize global variables
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for i in range(iterations):
                #compute teh gradients and the costs
                grads, total_cost, content_cost, style_cost = self.compute_grads(generated_image)
                # Apply the gradients manually
                sess.run(apply_grads, feed_dict={grads_placeholder: grads})
                #check whether the current cost is the best cost,if its not, update the variables (best_cost, best_image)
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_image = sess.run(generated_image)
                #print the costs every step iterations 
                if step is not None:
                    if i % step == 0:
                        print("Cost at iteration {}: {}, content {}, style {}".format(i, curr_total_cost, content_cost, style_cost))
            #after all the iterations, return the best cost and the best image
            return best_image, best_cost
