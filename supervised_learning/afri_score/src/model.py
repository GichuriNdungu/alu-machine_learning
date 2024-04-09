# import the right modules

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_model(input_shape, lstm_units, dense_units, n_dense_layers=1, activation='relu', output_activation='sigmoid', learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=input_shape))

    for _ in range(n_dense_layers):
        model.add(Dense(dense_units, activation=activation))

    model.add(Dense(1, activation=output_activation))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    return model
