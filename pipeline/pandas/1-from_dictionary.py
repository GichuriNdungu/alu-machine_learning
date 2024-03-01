#!/usr/bin/env python3
import pandas as pd 

dictionary = {'First': [0.0, 0.5,0.1,1.5], 'second':['one', 'two', 'three', 'four']}
df = pd.DataFrame(dictionary,index=['A', 'B', 'C', 'D'])
print(df)