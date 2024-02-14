#!/usr/bin/env python3
'''function that calculates the precision for each class in a confusion matrix'''
import numpy as np


def precision(confusion):
    '''args: confusion
    return: precision'''
    classes = confusion.shape[0]
    precision = np.zeros(classes)
    for i in range(classes):
        precision[i] = confusion[i][i] / np.sum(confusion[:, i])
    return precision
