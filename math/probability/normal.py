#!/usr/bin/env python3

'''class Normal that represents
 a normal distribution
 args: mean, data, stddev
 return: standard dev, mean'''


class Normal:
    '''class constructor'''

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                mean = sum(data)/len(data)
                self.mean = mean
                squared_diff_sum = 0
                for i in data:
                    squared_diff = (i-self.mean)**2
                    squared_diff_sum += squared_diff
                starndard_dev = (squared_diff_sum/len(data)) ** 0.5
                self.stddev = starndard_dev
