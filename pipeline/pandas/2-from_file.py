#!/usr/bin/env python3
import pandas as pd 

def from_file(filename, delimeter):
    df = pd.read_csv(filename, delimiter=delimeter)
    return df