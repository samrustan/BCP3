import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as f:
    driving_data = csv.reader(f)
    for line in driving_data: 
        lines.append(line)

images = []
steering_angles = []

for line in lines:
    src_path = line[0]
    filename = src_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    steering_angle = float(line[3])
    steering_angles.append(steering_angle)

def batch_generator(image_paths, angles, batch_size=128):
    X, y = ([],[])
    while True:
        for i in range(len(angles)):
            image = cv2.imread(image_paths[i])
            angle = angles[i]
            X.append(image)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
        

# convert to numpy arrays
X_train = np.array(images)
y_train = np.array(steering_angles)

# run through NN based on NVIDIA/JShannon model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda
from keras.layers import Convolution2D, ELU
from keras.regularizers import l2

model = Sequential()
# Normalize
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160,320,3)))

# Three 5x5 conv layers; output depth: 24,36,48; stride: 2x2 valid
model.add(Convolution2D(24,5,5,subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36,5,5,subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48,5,5,subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Two 3x3 conv layers; output_depth:64,64
model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64,3,3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Fully Connected
model.add(Flatten())

# Three FC layers output_depth: 100, 50, 10
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())

# FC output layer
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
#model.fit_generator(batch_generator(X_train, y_train, batch_size=BATCH_SIZE),
#                                    samples_per_epoch=X_train.shape[0]*2,
#                                    nb_epoch=NB_EPOCH,
      

print("saving model...")

model.save('model.h5')

print("model saved!")
