import numpy as np
import os
import sys
import pandas as pd
from skimage.io import imread
import cv2

def getData():

    data = pd.read_csv("./data/driving_log.csv", header = 0)
    data.columns = ('center','left','right','steering','throttle','brake','speed')

    center = data[['center','steering']].copy()

    left = data[['left', 'steering']].copy()
    left.loc[:, 'steering'] += 0.2

    right = data[['right', 'steering']].copy()
    right.loc[:, 'steering'] -= 0.2

    image_paths = pd.concat([center.center, left.left, right.right]).str.strip()
    angles = pd.concat([center.steering, left.steering, right.steering])

    image_paths = image_paths.as_matrix()
    angles = angles.as_matrix()

    return image_paths, angles

def doData(image_path, angle, display=False):

    images = []
    angles = []
    image = imread('./data/' + image_path)
    resized = cv2.resize(image,(200,66))
    fresized = cv2.flip(resized, 1)
    fangle = angle * -1
    images.append(resized)
    images.append(fresized)
    angles.append(angle)
    angles.append(fangle)

    if display:
        #cv2.imshow('image', image)
        cv2.imshow('resized', resized)
        cv2.waitKey(0)
        cv2.imshow('fresized', fresized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return images, angles

def preprocess_image(image):

    return image

def generator(image_paths, angles, batch_size=128):
    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])
    while True:
        for i in range(len(angles)):
            image = cv2.imread(image_paths[i])
            angle = angles[i]
            image = preprocess_image(image)
            X.append(images)
            y.append(angles)


        yield (np.array(X), np.array(y))

