#!/usr/bin/env python3
'''function that builds, trains, and saves a neural network model
  in tensorflow using adam optimization, mini_gradient descent,
 learning rate decay, and batch normalization'''
import numpy as np
import tensorflow as tf
def shuffle_data(X, Y):
    '''args: X, y
    return: x_shuffle, y_shuffle'''
    zipped = list(zip(X, Y))
    np.random.shuffle(zipped)
    x_shuffle, y_shuffle = zip(*zipped)
    x_shuffle = np.array(x_shuffle)
    y_shuffle = np.array(y_shuffle)
    return x_shuffle, y_shuffle

def calculate_loss(y, y_pred):
    """
    Method to calculate the cross-entropy loss
    of a prediction
    Args:
        y: input data type label in a placeholder
        y_pred: type tensor that contains the DNN prediction

    Returns:

    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
def calculate_accuracy(y, y_pred):
    """
    method to calculate the accuracy of a prediction in a DNN
    Args:
        y: input data type label in a placeholder
        y_pred: type tensor that contains the DNN prediction

    Returns: Prediction accuracy

    """
    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
def create_layer(prev, n, activation):
    """
    method to create a TF layer
    Args:
        prev: tensor of the previous layer
        n: n nodes created
        activation: activation function

    Returns: Layer created with shape n

    """
    # Average number of inputs and output connections.
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode='FAN_AVG')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name='layer')
    return layer(prev)
def create_batch_norm_layer(prev, n, activation):
    '''args: prev: is the activated output of the previous layer
            n: number of nodes in the layer to be created
            activation: activation function that
              should be used on the output of the layer
            returns:tensor of the activated output of the layer'''
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None,
                        kernel_initializer=initializer)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    offset = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    epsilon = 1e-8
    normalization = tf.nn.batch_normalization(
        x_prev, mean, variance, offset, scale, epsilon)
    return activation(normalization)
def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Forward propagation method using TF
    Args:
        x: Input data (placeholder)
        layer_sizes: type list are the n nodes inside the layers
        activations: type list with the activation function per layer

    Returns: Prediction of a DNN

    """
    layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i != len(layer_sizes) - 1:
            layer = create_batch_norm_layer(layer,
                                            layer_sizes[i],
                                            activations[i])
        else:
            layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''args:loss;loss of the net
            alpha: learning rate
            beta1: momentum weight
            beta2: rmsprop weight
            return:rmsprop optimization operation'''
    optimizer = tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon=epsilon).minimize(loss)
    return optimizer
def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''args: alpha: initial learning rate
            decay_rate: weight to determin the rate of decay
            global_step: number of passes of gradient descent that have lapsed
            decay_step: number of passes of gradient descent before decay
            returns: updated alpha after each update'''
    return tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
def model(Data_train, Data_valid, layers,
           activations, alpha=0.001, beta1=0.9,
             beta2=0.999, epsilon=1e-8, decay_rate=1,
               batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):