# import the right modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.preprocessing import MinMaxScaler

df =    pd.read_csv('bank.csv', sep=';')
print(df.head(4))

# Select independent and dependent variables 
X = df[['job', 'marital', 'education', 'default', 'housing','duration', 'loan','contact','day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']]
Y = df[["y"]]
print(X, Y)
def select_features(X, Y, df,missing_threshold):
    '''drop columns with too many missing values'''
    clf = RandomForestClassifier()
    clf.fit(X, Y)
    importances = clf.feature_importances_
    columns_to_drop = X.columns[importances < 0.05]
    X_dropped = X.drop(columns_to_drop, axis =1)
    X_dropped = X_dropped.loc[:, X_dropped.isna().mean() < missing_threshold]
    return X_dropped
def encode_data(df):
    '''encode categorical data using label encoder'''
    le = LabelEncoder()
    for category in df.columns[df.dtypes == object]:
        df.loc[:, category] = le.fit_transform(df[category])
    return df
def oversample(X, Y):
    '''Augment the target variable to prevent class bias'''
    ros = RandomOverSampler()
    X, Y = ros.fit_resample(X,Y)
    return Y
def scale_features(x):
    '''use minmax scaler to scale the input features'''
    
    ms=MinMaxScaler()
    x=ms.fit_transform(x)
def split_data_set()
X = encode_data(X)
print(select_features(X, Y,df, missing_threshold=0.5))