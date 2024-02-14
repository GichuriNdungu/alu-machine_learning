#!/usr/bin/env python3
'''function that creates a confusion matrix'''
import numpy as np


def create_confusion_matrix(labels, logits):
    '''args: labels, logits
    return: confusion matrix'''
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    for i in range(m):
        j = np.argmax(labels[i])
        k = np.argmax(logits[i])
        confusion[j][k] += 1
    return confusion
