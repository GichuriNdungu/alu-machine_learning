import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from model import create_model
from preprocessing import encode_data, oversample, scale_features, select_features, split_data, df
import pickle

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

df_encoded = encode_data(df)
X = df_encoded.drop('y', axis=1)
Y = df_encoded['y']
Y = Y.astype(int)
X_selected = select_features(X, Y, df_encoded, missing_threshold=0.5)
# oversampling 
X_resampled, Y_oversampled = oversample(X_selected, Y)
X_scaled = scale_features(X_resampled)
# split the dataset

x_train, x_test, x_val, y_train, y_test, y_val = split_data(X_scaled, Y_oversampled)
x_train_lstm = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test_lstm = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
x_val_lstm = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))

# Train the model

model_1 = create_model(input_shape=(x_train_lstm.shape[1], x_train_lstm.shape[2]), lstm_units=50, dense_units=10, n_dense_layers=3)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model_1.fit(x_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(x_val_lstm, y_test))
_, accuracy = model_1.evaluate(x_val_lstm, y_val)
# print model accuracy
print(f'Model Accuracy: {accuracy * 100:.2f}%')
# save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model_1, f)

