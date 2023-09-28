#!/usr/bin/env python3
'''Class Poisson that sets the lambtha
of a poisson distribution
Args = data
'''


class Poisson:
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

    def pmf(self, k):
        '''calculates the PMF of a number
        of successes'''
        if not isinstance(k, int):
            k = int(k)
        if k >= 0:
            e = 2.7182818285
            num = e**(-self.lambtha)*(self.lambtha**k)
            result = 1
            for i in range(1, k+1):
                result *= i
            pmf = num/result
            return pmf
        else:
            return 0

    def cdf(self, k):
        '''calculates the cumulative
        distribution fucntion of k successes'''
        if not isinstance(k, int):
            k = int(k)
        cdf_value = 0
        for i in range(k+1):
            pmf_i = self.pmf(i)
            cdf_value += pmf_i
        return cdf_value
