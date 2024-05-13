#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
    '''A function that creates a dataframe from a numpy array'''
    df = pd.DataFrame(array)
    return df
