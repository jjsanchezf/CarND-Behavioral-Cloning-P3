import os
import csv
import cv2
import json
import numpy as np
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Activation, Lambda, Flatten, Dense, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam


#initialize flags
load_model_for_retraining = False
use_3cams=True
use_flip=True
partial_data=1 #use a fraction of the total images
epochs=1
size_batch= 32

# Read in the image and flip if necessary
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
data_directory = '../../../opt/carnd_p3/data/IMG/'

driving_log_path = 'driving_log.csv'

with open(driving_log_path) as csvfile:
	print('reading provided driving log')
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

num_frames = len(samples)
print('received {} samples'.format(num_frames))


# Process the list so that we end up with training images and labels
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

# Also, in order to generate more samples, lets add the entries that
# have non-zero angles flipped and negate the angles
if use_flip:
    for i in range(num_frames):
        if X_train[i][1] != 0.0:
            X_train.append([X_train[i][0], -1.0 * X_train[i][1], 1]) # flip flag

num_frames = len(X_train)

# Split some of the training data into a validation dataset.
# First lets shuffle the dataset, as we added lots of non-zero elements to the end
np.random.shuffle(X_train)
print('Splitting driving log samples into training and validation samples')
training_samples, validation_samples =train_test_split(X_train, test_size=0.2)
print('{} training samples split'.format(len(training_samples)))
print('{} validation samples split'.format(len(validation_samples)))
# Define the generator function that can be used to get batches of samples

def generator(samples, batch_size=size_batch):
	adjustment = 0.08
	num_samples = int(len(samples)/partial_data)
	# As long as the generator is iterated, provide more batches
	while True:
		# Shuffle the samples before splitting out the batches
		np.random.shuffle(samples)

		# Loop over the samples to generate batches.
		for offset in range(0, num_samples, batch_size):

			# Get this batch of samples from the overall listing.
			batch_samples = samples[offset:offset+batch_size]

			# Load the three images for this sample, and add the steering angles for each.
			images = []
			steering_angles = []
			for batch_sample in batch_samples:
				# Load the image for this sample.
				filename = batch_sample[0]
				angle = batch_sample[1]
				flip = float(batch_sample[2])
				image = process_image(filename, flip)				

				# Add the image to the list for this batch.
				images.append(image)

				# Add the steering angle to the list for this batch.
				steering_angles.append(angle)			

			# Cast the batch into numpy arrays.
			X_data = np.array(images)
			y_data = np.array(steering_angles)

			# Yield the images shuffled, to obscure the augmentations.
			yield shuffle(X_data, y_data)

# Create the generators for our training and validation sets.
training_generator = generator(training_samples, batch_size=size_batch)
validation_generator = generator(validation_samples, batch_size=size_batch)
"""
Keras Model provided below.
"""
if not load_model_for_retraining:
	# Create the model based off of the Nvidia model structure.

	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
	model.add(Cropping2D(cropping=((70, 25), (0, 0))))
	model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(64, 3, 3, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(Dense(50))
	model.add(Dropout(0.1))
	model.add(Dense(10))
	model.add(Dense(1))
else:
	# Load the model that was previously trained.
	model = keras.models.load_model('./model_V3.h5')

# Create an ADAM optimizer.

optimizer = Adam()

# Use the ADAM optimizer and mean squared error for compilation.

model.compile(loss='mse', optimizer=optimizer)

# Train the model for X epochs.

model.fit_generator(
	training_generator,
	samples_per_epoch=int(len(training_samples)/partial_data),
	validation_data=validation_generator,
	nb_val_samples=int(len(validation_samples)/partial_data),
	nb_epoch=epochs,
)
model.save('model_V4.h5')

# Save the model when training session is finished.
model_json = model.to_json()
with open("./model_V4.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("./model_weights_V4.h5")
print("Saved model to disk")
