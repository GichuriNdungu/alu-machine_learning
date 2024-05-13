#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

#First conv layer

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Second conv layer

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Third conv layer

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Add Dense layer to conv layers
#first flatten the 3d feature maps coming in from the convolutional networks into 1d feature vectors
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Prepare data before feeding into convnet

batch_size = 20

# define augmentation for training

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

# define augmentation for testing (just rescale the images)

test_datagen = ImageDataGenerator(rescale=1./255)

# now Create a generator that will randomly generate augmented images from our data set

train_generator = train_datagen.flow_from_directory('C:/Users/user/Desktop/codes/alu-machine_learning/supervised_learning/input/Cats_and_dogs/training_set/training_set',
                                                     target_size=(150, 150),
                                                       batch_size=batch_size,
                                                         class_mode='binary')
sample_batch = next(train_generator)
input_images = sample_batch[0]
labels = sample_batch[1]
print('shape of input images', input_images.shape)
print('shape of labels', labels.shape)

# now define generator for the validation data set

validation_generator = test_datagen.flow_from_directory('C:/Users/user/Desktop/codes/alu-machine_learning/supervised_learning/input/Cats_and_dogs/test_set/test_set',
                                                     target_size=(150, 150),
                                                       batch_size=batch_size,
                                                         class_mode='binary')
# # now we can  use these generators to train our model
# # we might need to swith from CPU to GPU for more compute power, but we will first try with simply CPU

model.fit_generator(train_generator,
                    epochs=10, validation_data=validation_generator)
model.save_weights('../input/first_try.h5')