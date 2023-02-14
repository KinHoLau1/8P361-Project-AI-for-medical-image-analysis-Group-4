# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:05:42 2023

@author: kinho
"""

import os, os.path
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

#%%
#Find py-file directory
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#Add image folders to path
#py-file and image folders need to be in the same directory
train0DIR = os.path.join(__location__, 'train\\0')
train1DIR = os.path.join(__location__, 'train\\1')

#Determine the number of images in each folder
lenTrain0 = len([name for name in os.listdir(train0DIR) if os.path.isfile(os.path.join(train0DIR, name))])
lenTrain1 = len([name for name in os.listdir(train1DIR) if os.path.isfile(os.path.join(train1DIR, name))])

#%%
#Select k random images from each class
k = 5  
imagenrs0 = [random.randint(0, lenTrain0) for i in range(k)]
imagenrs1 = [random.randint(0, lenTrain1) for i in range(k)]

images0 = [mpimg.imread(os.path.join(train0DIR, os.listdir(train0DIR)[i])) for i in imagenrs0]
images1 = [mpimg.imread(os.path.join(train1DIR, os.listdir(train1DIR)[i])) for i in imagenrs1]

#Show images
fig = plt.figure(constrained_layout=True)

# create 2x1 subfigs
subfigs = fig.subfigures(nrows=2, ncols=1)
for row, subfig in enumerate(subfigs):
    if row == 0:
        subfig.suptitle('No metastases')
    else:
        subfig.suptitle('Metastases')

    # create 1x5 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=5)
    for col, ax in enumerate(axs):
        if row == 0:
            ax.imshow(images0[col])
        else:
            ax.imshow(images1[col])
            
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)