#!/usr/bin/env python3
'''function that calculates the F1 score
 for each class in a confusion matrix'''
import numpy as np


def f1_score(confusion):
    '''args: confusion
    return: f1 score'''
    classes = confusion.shape[0]
    f1 = np.zeros(classes)
    precision = precision(confusion)
    sensitivity = sensitivity(confusion)
    for i in range(classes):
        f1[i] = 2 * (precision[i] * sensitivity[i]) / \
            (precision[i] + sensitivity[i])
    return f1
