#!/usr/bin/env python3
'''function that updates a variable using the
 gradient descent with momentum optimization technique'''


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''args: alpha: learning rate
            beta1: momentum weight
            var: variable to update
            grad:gradient descent of var
            v: previous first moment of var
            return:updated variable, new moment'''
    # predefine vo as 0
    vo = 0

    vt = v*beta1 + ((1-beta1)*grad)
    # non_bias = vt/(1-beta1**)
    new_value = var - alpha*vt
    v0 = vt
    return new_value, vt
