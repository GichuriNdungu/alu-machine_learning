#!/usr/bin/env python3
'''function that calculates the
specificity for each class in a confusion matrix'''
import numpy as np
def specificity(confusion):
    '''args: confusion
    return: specificity'''
    classes = confusion.shape[0]
    specificity = np.zeros(classes)
    for i in range(classes):
        true_negative = np.sum(np.delete(confusion, i, 0),
                               axis=0) - confusion[i][i]
        false_positive = np.sum(confusion[:, i]) - confusion[i][i]
        specificity[i] = true_negative / (true_negative + false_positive)
    return specificity