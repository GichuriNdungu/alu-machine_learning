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

    sdw = s * beta2 + (1-beta2)*(grad**2)
    dw_multplier = grad/np.sqrt((sdw)+epsilon)
    new_value = var - (alpha*dw_multplier)
    s = sdw
    return new_value, sdw
