#!/usr/bin/env python3
''''function to update the
 weights and biases of a neural net using gradient descent'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''args: weights: dictionary of the weights and biases of the neural net
       cache: dictionary of the outputs of each layer of the neural net
       alpha: learning rate
       lambtha: regularization parameter
       L: number of layers of the network
       Y: one-hot numpy.ndarray of shape (classes, m) that contains the
          correct labels for the data
       returns: nothing. updates the weights and biases of the network'''
    # get the number of data points
    m = Y.shape[1]
    # define the output from each layer
    out_err = cache['A' + str(L)] - Y
    # kictart backward propagation
    for layer in range(L, 0, -1):
        # prev_out represents the output from the previous layer,
        # which is the input of our current layer
        Prev_out = cache['A' + str(layer - 1)]
        # calculate the gradients of the weights and biases
        dw = (1/m) * np.dot(out_err, Prev_out.T)
        db = (1/m) * np.sum(out_err, axis=1, keepdims=True)
        # gradients of the regularization term
        dw_regularization = (lambtha/m) * weights['W'+str(layer)]
        # add the regularized weights to the derivative of the weights
        dw += dw_regularization
        # get the gradient of the previous layers activation
        dz_prev = np.dot(weights['W'+str(layer)].T,
                         out_err) * (Prev_out*(1-Prev_out))
        # update the weights and the biases
        weights['W' + str(layer)] -= alpha * dw
        weights['b' + str(layer)] -= alpha * db

        # set the output for the next iteration
        dz_prev = out_err
