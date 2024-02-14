#!/usr/bin/env python3
'''function that calculates the sensitivity for each class in a confusion matrix'''
import numpy as np


def sensitivity(confusion):
    '''args: confusion
    return: sensitivity'''
    classes = confusion.shape[0]
    sensitivity = np.zeros(classes)
    for i in range(classes):
        sensitivity[i] = confusion[i][i] / np.sum(confusion[i])
    return sensitivity
