
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.preprocessing import MinMaxScaler
import pickle


df = pd.read_csv('bank.csv', sep=';')

# Select independent and dependent variables
X = df[['job', 'marital', 'education', 'default', 'housing', 'duration', 'loan',
        'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']]
Y = df[["y"]]

def select_features(X, Y, df, missing_threshold):
    '''drop columns with too many missing values'''
    clf = RandomForestClassifier()
    clf.fit(X, Y)
    importances = clf.feature_importances_
    columns_to_drop = X.columns[importances < 0.02]
    X_dropped = X.drop(columns_to_drop, axis=1)
    X_dropped = X_dropped.loc[:, X_dropped.isna().mean() < missing_threshold]
    return X_dropped


def encode_data(df):
    '''encode categorical data using a 
    label encoder and save the encoders'''
    le_dict = {}
    for category in df.columns[df.dtypes == object]:
        le = LabelEncoder()
        df.loc[:, category] = le.fit_transform(df[category])
        le_dict[category] = le
    # save the dictionary of encoders
    with open('models/encoder_dict.pkl', 'wb') as f: 
        pickle.dump(le_dict, f)
    return df


def oversample(X, Y):
    '''Augment the target variable to prevent class bias'''
    ros = RandomOverSampler()
    X_resampled, Y_resampled = ros.fit_resample(X, Y)
    return X_resampled, Y_resampled


def scale_features(x):
    '''use minmax scaler to scale the input features'''

    ms = MinMaxScaler()
    return ms.fit_transform(x)


def split_data(x, y, test_size=0.3, val_size=0.5, random_state=4):
    '''split the dataset into training, validation and testing sets'''
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    x_test, x_val, y_test, y_val = train_test_split(
        x_temp, y_temp, test_size=val_size, random_state=random_state)
    return x_train, x_test, x_val, y_train, y_test, y_val
