#!/usr/bin/env python3
'''Class Poisson that sets the lambtha
of a poisson distribution
Args = data
'''


class Exponential:
    '''class constructor'''

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            self.lambtha = lambtha
        elif type(data) != list:
            raise TypeError('data must be a list')
        elif len(data) < 2:
            raise ValueError('data must contain multiple values')
        else:
            sum_of_data = sum(data)
            self.lambtha = float(sum_of_data/len(data))
        if self.lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        else:
            pass
