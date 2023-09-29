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

    def z_score(self, x):
        '''calculates the z score of a given value x'''
        z_score = (x - self.mean)/self.stddev
        return z_score

    def x_value(self, z):
        '''calculates the value of a given z-score'''
        x_value = (z*self.stddev) + self.mean
        return x_value

    def pdf(self, x):
        '''calculates the pdf of a value x'''
        pi = 3.1415926536
        e = 2.7182818285
        coeffecient = 1/(self.stddev * ((2*pi)**0.5))
        exponent = -0.5 * (((x - self.mean)/self.stddev) ** 2)
        pdf = coeffecient * e**exponent
        return pdf

    def cdf(self, x):
        '''calculates the cdf of a value x'''
        mean = self.mean
        stddev = self.stddev
        pi = 3.1415926536
        value = (x - mean) / (stddev * (2 ** (1 / 2)))
        # estimate the erf using taylor series expansion
        erf = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        erf = erf - ((value ** 7) / 42) + ((value ** 9) / 216)
        # calculate the cdf from the estimated erf
        erf *= (2 / (pi ** (1 / 2)))
        cdf = (1 / 2) * (1 + erf)
        return cdf
