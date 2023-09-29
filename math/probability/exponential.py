#!/usr/bin/env python3

'''class Poisson that sets the lambtha
of a poisson distribution
Args = data
'''


class Exponential:
    '''class constructor'''

    def __init__(self, data=None, lambtha=1.):
        try:
            if data is None:
                if lambtha < 1:
                    raise ValueError("lambtha must be a positive value")
                else:
                    self.lambtha = float(lambtha)
            else:
                if type(data) is not list:
                    raise TypeError("data must be a list")
                elif len(data) < 2:
                    raise ValueError("data must contain multiple values")
                else:
                    lambtha = float(len(data) / sum(data))
                    self.lambtha = lambtha
        except Exception as e:
            print(str(e))

    def pdf(self, x):
        '''calculates the pdf of an 
        exponential distribution'''
        try:
            if not isinstance(x, int):
                x = int(x)
            if x >= 0:
                e = 2.7182818285
                lambtha = self.lambtha
                pdf = lambtha * (e**(-self.lambtha * x))
                return pdf
            else:
                return 0
        except Exception as e:
            print(str(e))
