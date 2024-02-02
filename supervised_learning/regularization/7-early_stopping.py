#!/usr/bin/env python3
'''function that determines if you should stop gradient descent early'''


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''args: 
            cost: current validation cost
            opt_cost: Lowest recorded validation cost of the nn
            threshold: threshold upon which if the validation cost
                        does not go under, early stopping is triggered
            patience count: specified training epochs, if the validation cost does not
                        meet the threshold after this patience period, stop
            count: count the number of times the threshold has not been met'''
    if cost < opt_cost - threshold:
        # reset the count, the model is still learning
        count = 0
        return False, count
    else:
        # increase the count, the model is beyond tolerance level
        count += 1
        return False, count
    if count >= patience:
        return True, count
