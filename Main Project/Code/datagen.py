# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:58:30 2023

@author: kinho
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, preprocessing=False,
                        pca_r=[],pca_g=[],pca_b=[]):
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
    preprocessing : Boolean determining whether or not to apply dimensionality reduction
    The default is False
    pca_r, pca_g, pca_b: IPCA models to use if dimensionality reduction is applied
    The default is []
    '''
    # dataset parameters
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')


    RESCALING_FACTOR = 1./255

    # instantiate data generators
    # apply dimensionality reduction if preprocessing is not False
    if preprocessing == False:
        datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
    else:
        datagen = ImageDataGenerator(rescale=RESCALING_FACTOR, 
                                     preprocessing_function=dim_red(pca_r,pca_g,pca_b))

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=train_batch_size,
                                            class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=val_batch_size,
                                            class_mode='binary')

    return train_gen, val_gen

# preprocessing function needs to be a Functor to pass parameters (IPCA objects)
class dim_red():
    def __init__(self, pca_r, pca_g, pca_b):
        self.pca_r = pca_r
        self.pca_g = pca_g
        self.pca_b = pca_b
        
    def __call__(self, img):
        # split images into three color channels and flatten
        img_r = img[:,:,0].reshape(1,-1)
        img_g = img[:,:,1].reshape(1,-1)
        img_b = img[:,:,2].reshape(1,-1)
        
        # apply dimensionality reduction to images
        img_rt = self.pca_r.transform(img_r)
        img_gt = self.pca_g.transform(img_g)
        img_bt = self.pca_b.transform(img_b)
        
        # transform data back into original space
        img_rinv = self.pca_r.inverse_transform(img_rt).reshape(IMAGE_SIZE,IMAGE_SIZE)
        img_ginv = self.pca_g.inverse_transform(img_gt).reshape(IMAGE_SIZE,IMAGE_SIZE)
        img_binv = self.pca_b.inverse_transform(img_bt).reshape(IMAGE_SIZE,IMAGE_SIZE)
        
        # merge color channels into RGB image
        img_rec = cv2.merge((img_rinv,img_ginv,img_binv))
        return img_rec