#!/usr/bin/env python3
'''function that calculates the F1 score
 for each class in a confusion matrix'''
import numpy as np
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    '''args: confusion
    return: f1 score'''
    classes = confusion.shape[0]
    f1 = np.zeros(classes)
    precision_values = precision(confusion)
    sensitivity_values = sensitivity(confusion)
    for i in range(classes):
        f1[i] = 2 * (precision_values[i] * sensitivity_values[i]) / \
            (precision[i] + sensitivity[i])
    return f1
