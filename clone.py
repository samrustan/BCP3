import csv
import cv2
import numpy as np

lines = []
with open('./data_new/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader: 
        lines.append(line)

images = []
measurements = []

for line in lines:
    src_path = line[0]
    filename = src_path.split('/')[-1]
    current_path = './data_new/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
print(current_path)
# run through NN --simplified 
# convert to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

# run through simple NN
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

print("saving model...")

model.save('model.h5')


