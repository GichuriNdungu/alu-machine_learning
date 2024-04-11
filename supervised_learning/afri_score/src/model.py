# import the right modules

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing import encode_data, oversample, scale_features, select_features, split_data, df
import pickle


def create_model(input_shape, lstm_units, dense_units, n_dense_layers=1, activation='relu', output_activation='sigmoid', learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=input_shape))

    for _ in range(n_dense_layers):
        model.add(Dense(dense_units, activation=activation))

    model.add(Dense(1, activation=output_activation))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    return model

def prepare_data(df):
    df_encoded = encode_data(df)
    X = df_encoded.drop('y', axis=1)
    Y = df_encoded['y']
    Y = Y.astype(int)
    X_selected = select_features(X, Y, df_encoded, missing_threshold=0.5)
    X_resampled, Y_oversampled = oversample(X_selected, Y)
    print(X_resampled)
    X_scaled = scale_features(X_resampled)
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(X_scaled, Y_oversampled)
    x_train_lstm = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test_lstm = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    x_val_lstm = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
    return x_train_lstm, y_train, x_val_lstm, y_val

def train_and_save_model(x_train_lstm, y_train, x_val_lstm, y_val):
    model_1 = create_model(input_shape=(x_train_lstm.shape[1], x_train_lstm.shape[2]), lstm_units=50, dense_units=10, n_dense_layers=3)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model_1.fit(x_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(x_val_lstm, y_val))
    _, accuracy = model_1.evaluate(x_val_lstm, y_val)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model_1, f)

# Use the functions
x_train_lstm, y_train, x_val_lstm, y_val = prepare_data(df)
train_and_save_model(x_train_lstm, y_train, x_val_lstm, y_val)
