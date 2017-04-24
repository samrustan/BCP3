import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

samples = []
i = 0
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if i > 0 :
            samples.append(line)
        i = i+1
	
def preprocess_image(img):
    new_img = img[50:140,:,:]
    new_img = cv2.resize(new_img ,(200, 66))
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

from sklearn.model_selection import train_test_split
train_samples, validation_samples= train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    image_path = './data/IMG/'
    while 1: # Loop forever so the generator never terminates
        small_batch_size = 32
        shuffle(samples)
        for offset in range(0, num_samples, small_batch_size):
            batch_samples = samples[offset:offset+small_batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:

                for i in range(3):
                    source_path = './data/IMG/'+batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)
                    processed_image = preprocess_image(image)
                    steering_correction = 0.2 
                    steering_angle = float(batch_sample[3])
                    if i == 1:
                         steering_angle += steering_correction
                    elif i == 2:
                         steering_angle -= steering_correction

                    images.append(processed_image)
                    steering_angles.append(steering_angle)
                    images.append(cv2.flip(processed_image,1))
                    steering_angles.append(steering_angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(steering_angles)

            yield sklearn.utils.shuffle(X_train, y_train)
	
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size= 128)
validation_generator = generator( validation_samples, batch_size=128)


# imports for model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout 
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, Callback
from keras.regularizers import l2

# model based on NVIDIA End-to-End model
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66,200,3)))

# 3 Convolutional layers with 2x2 stride, ELU activation`
model.add(Convolution2D(24, 5, 5,subsample = (2,2), border_mode='valid', activation="elu", W_regularizer=l2(0.001)))
model.add(Convolution2D(36, 5, 5,subsample = (2,2), border_mode='valid', activation="elu", W_regularizer=l2(0.001)))
model.add(Convolution2D(48, 5, 5,subsample = (2,2), border_mode='valid', activation="elu" , W_regularizer=l2(0.001)))

# 2 more convolutional layers
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="elu", W_regularizer=l2(0.001)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="elu", W_regularizer=l2(0.001)))

model.add(Flatten())

# 3 Fully Connected layers and output layer
model.add(Dense(100, activation="elu", W_regularizer=l2(0.001)))
model.add(Dense(50, activation="elu", W_regularizer=l2(0.001)))
model.add(Dense(10, activation="elu", W_regularizer=l2(0.001)))
model.add(Dense(1))

# model compiled and model checkpoints added after each epoch
model.compile(loss='mse', optimizer='adam')
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples) * 6,
                    validation_data=validation_generator, nb_val_samples=len(validation_samples)*6,
                    nb_epoch = 5, callbacks=[checkpoint])

model.save('model.h5') 
