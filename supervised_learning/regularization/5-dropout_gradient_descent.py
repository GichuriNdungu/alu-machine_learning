#!/usr/bin/env python3
'''defines a function that updates
 the weights of a neural network  with dropout regularization
 using gradient descent'''
import numpy as np
def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''update the weights and biases of a neural net with dropout regularization'''