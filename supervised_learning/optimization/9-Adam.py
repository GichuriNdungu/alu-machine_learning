#!/usr/bin/env python3
'''function that updates variables using adam optimization algorithm'''
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''args: alpha: learning rate
            beta1: momentum weight for moment 1
            beta2: rmsprop weight for moment 2
            epilson: value to prevent 0-division
            var: variable to update
            grad: gradient descent of var
            v: previous first moment of var
            s: previous second moment of var
            t: timestep used for bias correction
            return: updated variable, new moment'''
    vt = v*beta1 + ((1-beta1)*grad)
    vdw = vt/(1-beta1**t)
    Sdv = (beta2 * s) + ((1 - beta2) * grad ** 2)
    non_bias_sdv = Sdv/(1-beta2**t)
    new_variable = var - (alpha*(vdw/np.sqrt((non_bias_sdv) + epsilon)))
    return new_variable, vdw, non_bias_sdv
