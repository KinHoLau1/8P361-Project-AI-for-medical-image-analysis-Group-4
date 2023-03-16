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

from datagen import get_pcam_generators
    
#%%
# load IPCA models
parent = dirname(dirname(abspath(__file__)))
folder = parent + "\IPCA Models\\"
pca_r = pk.load(open(folder + "pca_r_all.pkl",'rb'))
pca_g = pk.load(open(folder + "pca_g_all.pkl",'rb'))
pca_b = pk.load(open(folder + "pca_b_all.pkl",'rb'))
# get the data generators (with real time dimensionality reduction)
train_gen, val_gen = get_pcam_generators('C:\8P361',preprocessing=True,
                                         pca_r=pca_r,pca_g=pca_g,pca_b=pca_b)
#%%
import matplotlib.pyplot as plt
plt.imshow(train_gen[0][0][5])