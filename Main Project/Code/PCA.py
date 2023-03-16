# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:02:22 2023

@author: kinho
"""

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

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA
import cv2
import random
import pickle as pk

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

def Inc_PCA(data,com=None):
    pca_r = IncrementalPCA(n_components=com)
    pca_g = IncrementalPCA(n_components=com)
    pca_b = IncrementalPCA(n_components=com)
    for i in range(len(data)):
        batch = data[i][0]
        samples = len(batch)
        r_batch = np.reshape(batch[:,:,:,0],(samples,-1))
        g_batch = np.reshape(batch[:,:,:,1],(samples,-1))
        b_batch = np.reshape(batch[:,:,:,2],(samples,-1))
        pca_r.partial_fit(r_batch)
        pca_g.partial_fit(g_batch)
        pca_b.partial_fit(b_batch)
        print("Batch",i+1,"done")
    return pca_r,pca_g,pca_b
    
#%%    
# get the data generators
train_gen, val_gen = get_pcam_generators('C:\8P361',10000,10000)
#%%
pca_r,pca_g,pca_b = Inc_PCA(train_gen)
#%%
for i in range (5):
    batch = random.randrange(len(train_gen))
    n = random.randrange(len(train_gen[batch][0]))
    img = train_gen[batch][0][n]
    img_r = img[:,:,0].reshape(1,-1)
    img_g = img[:,:,1].reshape(1,-1)
    img_b = img[:,:,2].reshape(1,-1)

    img_rt = pca_r.transform(img_r)
    img_gt = pca_g.transform(img_g)
    img_bt = pca_b.transform(img_b)
    
    img_rinv = pca_r.inverse_transform(img_rt).reshape(IMAGE_SIZE,IMAGE_SIZE)
    img_ginv = pca_g.inverse_transform(img_gt).reshape(IMAGE_SIZE,IMAGE_SIZE)
    img_binv = pca_b.inverse_transform(img_bt).reshape(IMAGE_SIZE,IMAGE_SIZE)
    img_rec = cv2.merge((img_binv,img_ginv,img_rinv))
    
    pair = np.concatenate((img, img_rec), axis=1)
    plt.figure(figsize=(4,2))
    plt.imshow(pair)
    plt.show()
#%%
pca = [pca_r,pca_g,pca_b]
plt.figure(figsize=[7, 10])
color = ['red','green','blue']
for i in range(len(pca)):
    exp_var = pca[i].explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)
    cum_exp_var = np.insert(cum_exp_var,0,0)
    com = pca[i].n_components_

    plt.step(range(1, com+2), cum_exp_var, where='mid',
         label='Cumulative explained variance', color=color[i])

plt.ylabel('Explained variance percentage')
plt.xlabel('Principal component index')
plt.legend(loc='right')
plt.title("PCA")
plt.tight_layout()
#%%
pk.dump(pca_r, open("pca_r_all.pkl","wb"))
pk.dump(pca_g, open("pca_g_all.pkl","wb"))
pk.dump(pca_b, open("pca_b_all.pkl","wb"))
#%%
pca_r.n_components_