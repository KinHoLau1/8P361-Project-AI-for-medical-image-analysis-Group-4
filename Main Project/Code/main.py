# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:25:45 2023

@author: kinho
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:02:22 2023

@author: kinho
"""

import pickle as pk
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

def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, IMAGE_SIZE=96):

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
# load IPCA models
parent = dirname(dirname(abspath(__file__)))
ipca_folder = parent + "\IPCA Models\\"

# select retained variance
ret_var = '90'
#ret_var = '80'
#ret_var = '70'
#ret_var = '60'

pca_r = pk.load(open(ipca_folder + "pca_r_"+ret_var+".pkl",'rb'))
pca_g = pk.load(open(ipca_folder + "pca_g_"+ret_var+".pkl",'rb'))
pca_b = pk.load(open(ipca_folder + "pca_b_"+ret_var+".pkl",'rb'))
# get the data generators (with real time dimensionality reduction)
train_gen, val_gen = get_pcam_generators('C:\8P361',1024,1024,preprocessing=True,
                                         pca_r=pca_r,pca_g=pca_g,pca_b=pca_b)
#%% build model
model = get_model()

# save the model and weights
model_folder = parent + "\CNN Models\\"
model_name = model_folder + 'IPCA_'+ret_var+'_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)

# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
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