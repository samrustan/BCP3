# Udacity Self-Driving Nano Degree Project #3
# Behaviorial Cloning

Project Overview:
---
The goal of this project is to use a vehicle simulator to collect data, train a neural network model, then use that model to drive the vehicle simulation autonomously. 

Summary of project goals: 

* Use the simulator to collect driving behavior data
* Build a convolution neural network using the Keras framework that learns from image and steering angle data
* Feed that data to the model by making use of a Python generator
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

## Rubric Points

#### Files Submitted & Code Quality

#### 1. Submission includes:
This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md 

#### 2. Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture
---
The model used in this project is based on the model architecture published by Nvidia for training their self-driving car. The Nvidia "End-to-End" architecture is available [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The following shows the model layers of the Nvidia model. 

<img src="./images/nVidia_model.png?raw=true" width="400px">

The model used in this project is summarized here:
* Data normalization using a Keras Lambda function such that all pixels values range from -1 to 1
* Three 5x5 convolution layers with a 2x2 stride (subsampling)
* Two 3x3 convolution layers 
* Three fully-connected layers with output depths of 100, 50, 10
* ELU activation is used for each convolution and fully connected layer
* L2 regularization (added after initial training runs)
* MSE loss and Adam optimizer

The training dataset was supplied by Udacity, additionally training data for lane-edge recovery was collected using the recording function of the simulator.

The model was initally trained with only center images and steering angles with no augmentation. A validation set of 20% was used for validation. Training the network over about 10 epochs showed that the model had a low loss (MSE) on the training set but a high loss on the validation set. This implied that the model was likely overfitting. Data augmentation was used in the generator to mediate the overfitting. After data augmentation, the losses showed improvement as the validation loss was now improved.  The training epochs were reduced to 5 as the loss seemed to oscillate for additional epochs.

### Training Strategy
---

### Data Collection and augmentation
The training dataset was supplied by Udacity, additionally training data for lane-edge recovery was collected using the recording function of the simulator.

Samples of the data:

Left:
<img src="./images/left.jpg?raw=true" width="200px">
Center:
<img src="./images/center.jpg?raw=true" width="200px">
Right:
<img src="./images/right.jpg?raw=true" width="200px">

A steering angle correction factor of 0.2 was added to the left and subtracted from the right image steering angles.  A flipped image and steeting angle were appended to the dataset to mediate the tendancy of a turning bias. The image was flipped to provide a mirror image.  To flip the angle, the original angle was simply multiplied by a -1. This flipping of the images and angles proved to have the most significant effect of the data augmentation techniques.

Original:
<img src="./images/original.png?raw=true" width="200px">
Flipped:
<img src="./images/flipped.png?raw=true" width="200px">

The image and steering angle data was split into a training and validation set. Of the original dataset, 20% was used exclusively for validation.  Data shuffling was used for both training and validation sets to further reduce overfitting.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn’t necessary.

### Data Processing

Image cropping was used after it was determined that much of the image data above the horizon was unnecessary as well as lower section of the image that contained part of the body of the vehicle. The images are then resized, since the Nvidia architecture requires image input with a size of 200px by 66px.

Original:
<img src="./images/original.png?raw=true" width="200px">
Cropped:
<img src="./images/cropped.png?raw=true" width="200px">

Images were converted from RGB to YUV, this was done mostly based on the Nvidia results.  The simulator outputs images in RBG as well as takes in RGB as input, and opencv (cv2) reads images in BGR.  Since the model converts this to YUV, the drive.py file was edited to convert the YUV images.

### Conclusion
---

Using the trained model, the vehicle was able to traverse a full track without going off the road.  Only one lap was recorded for video file size and rubric requirement purposes, but several laps were run without issue.  Using the Keras framework proved to be a method that would lend itself to fast prototyping a network.

This project is really where the rubber really hit the road for me.  I learned a great deal about how the quality of the data really affects the performance of the DNN model.  And not just performance based on MSE or loss, but the performance of the desired effect or outcome of the applied DNN.  There is a bit of common sense to this if you consider the anaology of trying to learn a language with a dictionary full of misspelled words.  For this project, watching the model perform on different datasets was most enlightening.
