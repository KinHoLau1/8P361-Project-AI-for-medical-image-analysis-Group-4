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

import os

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA
import cv2
import random
import pickle as pk
from os.path import dirname, abspath

from PCA import Inc_PCA

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    '''
    Generates Keras DirectoryIterator objects for training and validation
    datasets. These objects contain the images and their labels.

    Parameters
    ----------
    base_dir : path to directory containing datasets
    train_batch_size : The batch size of the training data object.
    The default is 32.
    val_batch_size : The batch size of the validation data object. 
    The default is 32.
    '''
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
    
#%%    
# get the data generators for IPCA
# batch size for IPCa will be equal to train_batch_size
train_gen, val_gen = get_pcam_generators('C:\8P361',10000,10000)
#%%
# train IPCA models
# number of retained components can not be higher than 9216 (number of features per channel)
# or train_batch_size
pca_r,pca_g,pca_b = Inc_PCA(train_gen)

# save trained IPCA objects
parent = dirname(dirname(abspath(__file__)))
folder = parent + "\IPCA Models\\"
pk.dump(pca_r, open(folder + "pca_r.pkl","wb"))
pk.dump(pca_g, open(folder + "pca_g.pkl","wb"))
pk.dump(pca_b, open(folder + "pca_b.pkl","wb"))