import os
import csv
import cv2
import json
import numpy as np
#import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from keras.models import Sequential, load_model, model_from_json
#from keras.layers import Activation, Lambda, Flatten, Dense, MaxPooling2D, Dropout, Cropping2D
#from keras.layers.convolutional import Conv2D
#from keras.optimizers import Adam


#initialize flags
use_3cams=True
use_flip=True


# Read in the imagee and flip in necessary
def process_image(filename, flip=0):
	image = cv2.imread(filename)
	image = image[...,::-1]
	if flip == 1:
		image = cv2.flip(image, 1)
	return image

"""
Data loading script provided below.
"""
# Load the driving log samples provided.
samples = []
#data_directory = '../../../opt/carnd_p3/data/IMG/'
data_directory = 'Data/track2/IMG/'

driving_log_path = 'driving_log_Track_2.csv'

with open(driving_log_path) as csvfile:
	print('reading provided driving log')
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

num_frames = len(samples)
print('received {} samples'.format(num_frames))


# Process this list so that we end up with training images and labels
if use_3cams:
    X_train = [("", 0.0, 0) for x in range(num_frames*3)]
    print(len(X_train))
    for i in range(num_frames):
        X_train[i*3] = (data_directory + samples[i][0],         # center image
                  float(samples[i][3]),  # center angle 
                  0)                              # dont flip
        X_train[(i*3)+1] = (data_directory + samples[i][1],       # left image
                  float(samples[i][3]) + 0.08,  # left angle 
                  0)                              # dont flip
        X_train[(i*3)+2] = (data_directory + samples[i][2],         # right image
                  float(samples[i][3]) - 0.08,  # right angle 
                  0)                              # dont flip
else:
    X_train = [("", 0.0, 0) for x in range(num_frames)]
    print(len(X_train))
    for i in range(num_frames):
        #print(i)
        X_train[i] = (samples[i][0],         # center image
                  float(samples[i][3]),  # center angle 
                  0)                              # dont flip

# Update num_frames as needed
num_frames = len(X_train)

# Also, in order to generate more samples, lets add entries twice for
# entries that have non-zero angles, and add a flip switch. Then when
# we are reading these, we will flip the image horizontally and 
# negate the angles
if use_flip:
    for i in range(num_frames):
        if X_train[i][1] != 0.0:
            X_train.append([X_train[i][0], -1.0 * X_train[i][1], 1]) # flip flag

num_frames = len(X_train)
print(num_frames)
# Split some of the training data into a validation dataset.
# First lets shuffle the dataset, as we added lots of non-zero elements to the end
#np.random.shuffle(X_train)

for sample in X_train:
	# Load the image for this sample.
	print(sample)
	filename = sample[0]
	print(filename)
	angle = sample[1]
	print(angle)
	flip = float(sample[2])
	image = process_image(filename, flip)				




