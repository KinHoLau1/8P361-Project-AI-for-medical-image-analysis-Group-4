# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:25:45 2023

@author: kinho
"""

from os.path import dirname, abspath
import os

# set working directory to location of py file
pypath = abspath(__file__)
dname = dirname(pypath)
os.chdir(dname)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from datagen import get_pcam_generators
from PCA import IPCA_load

def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, IMAGE_SIZE=96):
    '''
    Creates architecture for neural network

    Parameters
    ----------
    kernel_size, pool_size, first_filters, second_filters : parameters of the model layers
    IMAGE_SIZE : size of the images. The default is 96.

    Returns
    -------
    model : object describing the layers of the model
    '''

    # build the model
    model = Sequential()

    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPool2D(pool_size = pool_size))

    model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = pool_size))

    model.add(Conv2D(64, 6, activation = 'relu', padding = 'valid'))
    model.add(Conv2D(1, 1, activation = 'sigmoid', padding = 'valid'))
    model.add(Flatten())

    # compile the model
    model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

    return model
    
#%%
# load IPCa models
# select retained variance
#ret_var = '90'
ret_var = '80'
#ret_var = '70'
#ret_var = '60'

pca_r,pca_g,pca_b = IPCA_load(ret_var)
# data generators without PCA
#train_gen, val_gen = get_pcam_generators('C:\8P361',1024,1024)
# get the data generators (with real time dimensionality reduction)
train_gen, val_gen = get_pcam_generators('C:\8P361',1024,1024,preprocessing=True,
                                         pca_r=pca_r,pca_g=pca_g,pca_b=pca_b)
#%% build model
model = get_model()

# save the model and weights
parent = dirname(dirname(abspath(__file__)))
model_folder = parent + "\CNN Models\\"
model_name = model_folder + 'IPCA_'+ret_var+'_model'
# model_name = model_folder + 'fully_convolutional_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)

# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join(model_folder, 'Logs', 'IPCA_'+ret_var+'_model'))
# stop training early if validation loss stops decreasing
earlystopping = EarlyStopping(monitor="val_loss",mode="min", patience=1, restore_best_weights=True)
callbacks_list = [checkpoint, tensorboard, earlystopping]
#%% train model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=10,
                    callbacks=callbacks_list)