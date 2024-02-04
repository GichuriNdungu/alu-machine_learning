#!/usr/bin/env python3
'''function that calculates the weighted moving average of data'''


def moving_average(data, beta):
    '''args: data: data to calculate moving average
            beta: beta value for moving average calc
        return: list of moving averages for the data'''
    moving_avgs = []
    # declare vo as 0
    vo = 0
    for i, value in enumerate(data):
        vt = beta*vo + ((1-beta)*value)
        biased = vt/(1-beta**(i+1))
        moving_avgs.append(biased)
        vo = vt
    return moving_avgs
