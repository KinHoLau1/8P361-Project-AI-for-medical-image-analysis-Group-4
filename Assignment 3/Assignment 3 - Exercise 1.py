'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import model_from_json

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#%%
# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train')
     valid_path = os.path.join(base_dir, 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')

     return train_gen, val_gen


def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model


# get the model
model = get_model()

# get the data generators
train_gen, val_gen = get_pcam_generators('C:\8P361')

# save the model and weights
model_name = 'my_first_cnn_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]

#%%
# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)
#%%
MODEL_FILEPATH = 'my_first_cnn_model.json' 
MODEL_WEIGHTS_FILEPATH = 'my_first_cnn_model_weights.hdf5'

# load model and model weights
json_file = open(MODEL_FILEPATH, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(MODEL_WEIGHTS_FILEPATH)
#%%
# apply model to validation dataset
train_gen, val_gen = get_pcam_generators('C:\8P361')
# Create lists for storing the predictions and labels
predictions = []
labels = []

# Get the total number of labels in generator 
# (i.e. the length of the dataset where the generator generates batches from)
n = len(val_gen.labels)

# Loop over the generator
for data, label in val_gen:
    # Make predictions on data using the model. Store the results.
    predictions.extend(model.predict(data).flatten())

    # Store corresponding labels
    labels.extend(label)

    # We have to break out from the generator when we've processed 
    # the entire once (otherwise we would end up with duplicates). 
    if (len(label) <= val_gen.batch_size) and (len(predictions) == n):
        break
#%%
# ROC analysis
fpr, tpr, thresholds = roc_curve(labels, predictions)
auc_model = auc(fpr, tpr)
#%%
# visualisation
plt.plot([0,100],[0,100],'r--', label='Random classifier')
plt.plot(fpr*100, tpr*100, label='Model')
plt.xlabel('False positive rate (1-specificity) [%]')
plt.ylabel('True positive rate (sensitivity) [%]')
plt.title('ROC curve')
plt.xlim(0,100)
plt.ylim(0,100)
plt.legend()
plt.show()
print(auc_model)