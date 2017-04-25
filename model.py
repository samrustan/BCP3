import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# imports for model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, Callback
from keras.regularizers import l2


def preprocess_image(img):
    proc_img = img[50:140,:,:]
    proc_img = cv2.resize(proc_img ,(200, 66))
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2YUV)
    return proc_img

def generator(samples, batch_size=32):
    num_samples = len(samples)
    image_dir = 'data2/'
    steering_correction = 0.2
    left_cam_view = 1
    right_cam_view = 2

    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            image_dir = 'data2/'
            for batch_sample in batch_samples:
                # there are 3 camera views center, left, right
                for camera_view in range(3):
                    source_path = batch_sample[camera_view]
                    filename = source_path.split('/')[-1] 
                    IMG_dir = source_path.split('/')[-2] + '/'
                    data_type_dir = source_path.split('/')[-3] + '/'
                    current_path = image_dir + data_type_dir + IMG_dir + filename
                    image = cv2.imread(current_path)
                    processed_image = preprocess_image(image)
                    steering_angle = float(batch_sample[3])
                    # left camera view
                    if camera_view ==  left_cam_view:
                         steering_angle += steering_correction
                    # right camera view
                    elif camera_view == right_cam_view:
                         steering_angle -= steering_correction

                    # add a flipped image to the dataset to be generated
                    images.append(processed_image)
                    steering_angles.append(steering_angle)
                    images.append(cv2.flip(processed_image,1))
                    steering_angles.append(steering_angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(steering_angles)

#            yield sklearn.utils.shuffle(X_train, y_train)

# read in data from csv file
samples = []
lines = []
with open('data2/drive/driving_log.csv') as drive_data:
    reader = csv.reader(drive_data)
    for line in reader:
        samples.append(line)
with open('data2/recovery/driving_log.csv') as recovery_data:
    reader = csv.reader(recovery_data)
    for line in reader:
        samples.append(line)




# split into training and validation sets
train_samples, validation_samples= train_test_split(samples, test_size=0.2)
# create the training and validation generator objects
train_generator = generator(train_samples, batch_size= 32)
validation_generator = generator(validation_samples, batch_size=32)


# model based on NVIDIA End-to-End model
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66,200,3)))

# 3 Convolutional layers with 2x2 stride, ELU activation`
model.add(Convolution2D(24, 5, 5,subsample = (2,2), border_mode='valid', activation="elu", W_regularizer=l2(0.001)))
model.add(Convolution2D(36, 5, 5,subsample = (2,2), border_mode='valid', activation="elu", W_regularizer=l2(0.001)))
model.add(Convolution2D(48, 5, 5,subsample = (2,2), border_mode='valid', activation="elu" , W_regularizer=l2(0.001)))

model.add(Dropout(0.50))

# 2 more convolutional layers
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="elu", W_regularizer=l2(0.001)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation="elu", W_regularizer=l2(0.001)))

model.add(Flatten())

# 3 Fully Connected layers and output layer
model.add(Dense(100, activation="elu", W_regularizer=l2(0.001)))
#model.add(Dropout(0.50)
model.add(Dense(50, activation="elu", W_regularizer=l2(0.001)))
#model.add(Dropout(0.50)
model.add(Dense(10, activation="elu", W_regularizer=l2(0.001)))
#model.add(Dropout(0.50)
model.add(Dense(1))

# model compiled and model checkpoints added after each epoch
model.compile(loss='mse', optimizer='adam')
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples) * 6,
                    validation_data=validation_generator, nb_val_samples=len(validation_samples)*6,
                    nb_epoch = 5, callbacks=[checkpoint])

model.save('model.h5')
