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
    back = {}
    # kictart backward propagation
    for layer in range(L, 0, -1):
        # prev_out represents the output from the previous layer,
        # which is the input of our current layer
        prev_out = cache['A' + str(layer - 1)]
        
        if layer == L:
        #    get the error from the last layer
            back['dz{}'.format(layer)] = (cache['A{}'.format(layer)]-Y)
        else:
            # get the error from the preceeding layer from the right
            dz_prev = back['dz{}'.format(layer+1)]
            # get the output from the current layer
            A_current = cache['A'+ str(layer)]
            # calculate and update the error for the current layer
            back['dz{}'.format(layer)] = (np.matmul(curr_w.transpose(), dz_prev)* (A_current) * (1-A_current))
        # get the error in the current layer
        dz = back['dz{}'.format(layer)]
        dw = (1/m) * ((np.matmul(dz, prev_out.T))) + (lambtha * weights['W{}'.format(layer)])
        db = (1/m) * (np.sum(dz, axis=1, keepdims=True)) + (lambtha * weights['W{}'.format(layer)])
        curr_w = weights['W{}'.format(layer)]
        # update the weights and the biases
        weights['W' + str(layer)] -= alpha * dw
        weights['b' + str(layer)] -= alpha * db
