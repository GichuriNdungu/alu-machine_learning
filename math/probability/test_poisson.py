#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('F(9):', p1.cdf(9))

p2 = Poisson(lambtha=5)
print('F(9):', p2.cdf(9))


data = [1,2,3,4,5]
p2 = Poisson(data)
print('cdf:', p2.cdf(7))