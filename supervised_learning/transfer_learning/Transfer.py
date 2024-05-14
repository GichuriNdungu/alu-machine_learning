#!/usr/bin/env python3
'''script that performs transfer learning to train a new model 
Essentially takes over from conv.py'''

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# declare image dimensions: 

img_width, img_height = 150, 150
top_model_weights_path = '/..input/bottleneck_fc_model.h5'
train_data_dir = 'C:/Users/user/Desktop/codes/alu-machine_learning/supervised_learning/input/Cats_and_dogs/training_set/training_set'
validation_data_dir = 'C:/Users/user/Desktop/codes/alu-machine_learning/supervised_learning/input/Cats_and_dogs/test_set/test_set'
nb_train_samples = 5000
nb_validation_samples = 5000
epochs = 50
batch_size = 10

def save_bottleneck_features():
    '''function to save bottleneck features from the trained conv layers of convnet'''
    data_gen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 neural net
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = data_gen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode=None,  # generator yields just batches of data without labels
                                              shuffle=False)
    
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    np.save('C:/Users/user/Desktop/codes/alu-machine_learning/supervised_learning/transfer_learning/bottleneck_features_train.npy',
            bottleneck_features_train)
    
    generator = data_gen.flow_from_directory(validation_data_dir, 
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode = None,
                                             shuffle=False)
    bottle_neck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)

    np.save('C:/Users/user/Desktop/codes/alu-machine_learning/supervised_learning/transfer_learning/bottleneck_features_validation.npy',
            bottle_neck_features_validation)

def train_top_model():
    '''function that trains small, fully connected layer using the data saved from the bottleneck function above'''
    train_data = np.load('C:/Users/user/Desktop/codes/alu-machine_learning/supervised_learning/transfer_learning/bottleneck_features_train.npy')
    train_labels = np.array([0] * 2000 + [1] * 2000)

    validation_data = np.load('C:/Users/user/Desktop/codes/alu-machine_learning/supervised_learning/transfer_learning/bottleneck_features_validation.npy')
    validation_labels = np.array([0] * 1500 + [1] * 1500)

    model = Sequential()
    model.add(Flatten(input_shape = train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss = 'binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

save_bottleneck_features()
train_top_model()
                                                                          