#!/usr/bin/env python3

import numpy as np
import math as mt
class Poisson:
    def __init__(self, data=None, lambtha=1.):
        if data == None:
            self.lambtha = lambtha
        else:
            sum_of_data = sum(data)
            self.lambtha = float(sum_of_data/len(data))