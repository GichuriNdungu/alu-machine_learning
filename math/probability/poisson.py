#!/usr/bin/env python3
class Poisson:
    def __init__(self, data=None, lambtha=1.):
        if data == None:
            self.lambtha = lambtha
        else:
            sum_of_data = sum(data)
            self.lambtha = float(sum_of_data/len(data))
        if self.lambtha <= 0:
            raise ValueError('lambtha must be a positive value')
        else:
            pass
        