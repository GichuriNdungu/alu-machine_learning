import numpy as np
import math as mt
class Poisson:
    def __init__(self, data=None, lambtha=1.):
        sum_of_data = sum(data)
        self.lambtha = float(sum_of_data/len(data))

data = [1,2,3,4,5]
print(len(data))
p1 = Poisson(data=data)
print(p1.lambtha)
