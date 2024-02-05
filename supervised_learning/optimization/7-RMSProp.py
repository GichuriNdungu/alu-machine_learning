#!/usr/bin/env python3
'''function that updates variables using rmsprop algorithm'''
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''args: alpha: learning rate
            beta2: rmsprop weight
            epilson: value to prevent 0-division
            var: variable to update
            grad: gradient descent of var
            s: previous second moment of var
            return: updated variable, new moment'''
    Sdv = (beta2 * s) + ((1 - beta2) * grad ** 2)
    new_V = var - alpha * (grad / (Sdv ** (1 / 2) + epsilon))
    return new_V, Sdv
