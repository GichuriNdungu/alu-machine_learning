import numpy as np

class Neuron:
    def __init__(self, nx):
        '''nx refers to the number of input features'''
        if type(nx) != int:
            raise Exception('nx must be an integer')
        if nx < 1:
            raise Exception('nx must be a positive integer')
        self.w = np.random.normal()
        self.b = 0
        self.A = 0