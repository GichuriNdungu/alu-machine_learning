#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df['Close'] = df['Close'].fillna(method='ffill')
for column in ['Close', 'Low', 'Open']:
    df[column] = df[column].fillna(df['Close'])
for column in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[column] = df[column].fillna(0)
del df['Weighted_Price']
print(df.head())
print(df.tail())