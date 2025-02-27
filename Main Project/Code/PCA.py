# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:02:22 2023

@author: kinho
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA
import cv2
import random
import pickle as pk
from os.path import dirname, abspath
import os

from datagen import get_pcam_generators

# set working directory to location of py file
pypath = abspath(__file__)
dname = dirname(pypath)
os.chdir(dname)

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def Inc_PCA(data,com=[None,None,None]):
    '''
    Splits RGB images into their three color channels and performs
    Incremental PCA (IPCA) on each channel. The batch size used in the IPCA is
    equal to the batch size used in generating the Keras DirectoryIterator
    object passed to this function. The maximum number of components that can be
    kept is equal to the minimum between the bacth size and amount of features
    in the data (9216 for the PCAM dataset).
    
    Parameters
    ----------
    data : Keras DirectoryIterator object, this object contains the images and
    their labels. The IPCA models will be trained on these images.
    com : list of INTs, number of components to keep per color channel. Can not
    be higher than the batch size of the Keras DirectoryIterator object.
    Default for all channel is None, which sets it to the minimum between the 
    bacth size and amount of features in the data.
        
    Returns
    -------
    pca_r : Incremental PCA object trained on the red color channel
    pca_g : Incremental PCA object trained on the green color channel
    pca_b : Incremental PCA object trained on the blue color channel
    '''
    
    # define parameters for all models
    pca_r = IncrementalPCA(n_components=com[0])
    pca_g = IncrementalPCA(n_components=com[1])
    pca_b = IncrementalPCA(n_components=com[2])
    
    # train IPCA models on full dataset
    for i in range(len(data)):
        # select batch
        batch = data[i][0]
        samples = len(batch)
        # split images within batch into three color channels
        # and flatten
        r_batch = np.reshape(batch[:,:,:,0],(samples,-1))
        g_batch = np.reshape(batch[:,:,:,1],(samples,-1))
        b_batch = np.reshape(batch[:,:,:,2],(samples,-1))
        # partially train IPCA models per batch
        pca_r.partial_fit(r_batch)
        pca_g.partial_fit(g_batch)
        pca_b.partial_fit(b_batch)
        print("Batch {}/{} done".format(i+1,len(data)))
        
    return pca_r,pca_g,pca_b

def IPCA_load(ret_var):
    '''
    Loads pickle files containg IPCA objects

    Parameters
    ----------
    ret_var : string or int describing which IPCA objects to select

    Returns
    -------
    pca_r, pca_g, pca_b : IPCA objects for three color channels
    '''
    # get path to folder containing IPCA objects
    parent = dirname(dirname(abspath(__file__)))
    folder = parent + "\IPCA Models\\"
    
    # load IPCA objects
    pca_r = pk.load(open(folder + "pca_r_"+str(ret_var)+".pkl",'rb'))
    pca_g = pk.load(open(folder + "pca_g_"+str(ret_var)+".pkl",'rb'))
    pca_b = pk.load(open(folder + "pca_b_"+str(ret_var)+".pkl",'rb'))
    
    return pca_r,pca_g,pca_b

def IPCA_reconstruction(images,pca_r,pca_g,pca_b):
    '''
    Produces images with reduced number of components from origanal images

    Parameters
    ----------
    images : original images
    pca_r, pca_g, pca_b : IPCA objects

    Returns
    -------
    img_rec : recreation of images with reduced number of components

    '''
    samples = len(images)
    
    # split images into three color channels and flatten
    img_r = images[:,:,:,0].reshape(samples,-1)
    img_g = images[:,:,:,1].reshape(samples,-1)
    img_b = images[:,:,:,2].reshape(samples,-1)

    # apply dimensionality reduction to images
    img_rt = pca_r.transform(img_r)
    img_gt = pca_g.transform(img_g)
    img_bt = pca_b.transform(img_b)

    # transform data back into original space
    img_rinv = pca_r.inverse_transform(img_rt).reshape(samples,IMAGE_SIZE,IMAGE_SIZE)
    img_ginv = pca_g.inverse_transform(img_gt).reshape(samples,IMAGE_SIZE,IMAGE_SIZE)
    img_binv = pca_b.inverse_transform(img_bt).reshape(samples,IMAGE_SIZE,IMAGE_SIZE)
    
    img_rec = np.empty([samples,IMAGE_SIZE,IMAGE_SIZE,3])
    
    # merge color channels for each image
    for i in range(samples):
        img_rec[i] = cv2.merge((img_rinv[i],img_ginv[i],img_binv[i]))
        
    return img_rec
#%%
# prevent code execution when importing functions
if __name__ == '__main__':
    # get the data generators for IPCA
    # batch size for IPCa will be equal to train_batch_size
    train_gen, val_gen = get_pcam_generators('C:\8P361',5000,5000)
    #%%
    # train IPCA models
    # number of retained components can not be higher than 9216 (number of features per channel)
    # or train_batch_size
    pca_r,pca_g,pca_b = Inc_PCA(train_gen, [1263,787,1849])
    
    # save trained IPCA objects
    parent = dirname(dirname(abspath(__file__)))
    folder = parent + "\IPCA Models\\"
    pk.dump(pca_r, open(folder + "pca_r_80.pkl","wb"))
    pk.dump(pca_g, open(folder + "pca_g_80.pkl","wb"))
    pk.dump(pca_b, open(folder + "pca_b_80.pkl","wb"))
    #%%
    '''
    All code below is for testing/analysing the IPCA objects
    '''
    # load IPCa models
    pca_r_all,pca_g_all,pca_b_all = IPCA_load('all')
    pca_r,pca_g,pca_b = IPCA_load(80)

    #%%
    # view retained variance
    pca_list = [pca_r,pca_g,pca_b]
    for i in pca_list:
        print(np.sum(i.explained_variance_ratio_))
    #%% test IPCA models
    # use IPCA models to transform and reconstruct images
    nr_images = 5
    batch = random.randrange(len(train_gen))
    n = random.sample(range(len(train_gen[batch][0])),nr_images)
    img = train_gen[batch][0][n]
    
    img_rec = IPCA_reconstruction(img,pca_r,pca_g,pca_b)
    
    for i in range(nr_images):
        # show original and reconstructed images side-by-side
        f,ax = plt.subplots(1,2)
        ax[0].imshow(img[i])
        ax[0].axis('off')
        ax[0].set_title('original')
    
        ax[1].imshow(img_rec[i])
        ax[1].axis('off')
        ax[1].set_title('reconstruction')
    
    #%%
    # visualize cumulative variance explaned by each component
    pca = [pca_r_all,pca_g_all,pca_b_all]
    plt.figure(figsize=[7, 10])
    color = ['red','green','blue']
    # iterate over IPCA objects for each color channel
    for i in range(len(pca)):
        # extract and scale explaned variance ratios for each component
        exp_var = pca[i].explained_variance_ratio_ * 100
        # cumulatively add ratios
        cum_exp_var = np.cumsum(exp_var)
        cum_exp_var = np.insert(cum_exp_var,0,0)
        com = pca[i].n_components_
    
        plt.step(range(1, com+2), cum_exp_var, where='mid',
             label='Cumulative explained variance for the ' + color[i] + ' color channel', 
             color=color[i])
    
    plt.ylabel('Explained variance percentage')
    plt.xlabel('Principal component index')
    plt.legend(loc='right')
    plt.title("Cumulative explained variance")
    plt.tight_layout()
    
    # visualize explained variance per component per IPCA model
    for i in range(len(pca)):
        plt.figure()
        
        exp_var = pca[i].explained_variance_ratio_ * 100
        # plot without components with biggest explained variance
        plt.bar(range(4, pca[i].n_components_ + 1), exp_var[3:], align='center',
                color=color[i], width=100)
        
        plt.ylabel('Explained variance percentage')
        plt.xlabel('Principal component index')
        plt.title("Individual explained variance for the " + color[i] + ' color channel')
        plt.tight_layout()
    #%%
    # get index of max component to keep to retain target variance
    exp_var_r = pca_r_all.explained_variance_ratio_
    exp_var_g = pca_g_all.explained_variance_ratio_
    exp_var_b = pca_b_all.explained_variance_ratio_
    exp_var_list = [exp_var_r,exp_var_g,exp_var_b]
    
    # view explained variance ratio per component
    print(exp_var_r)
    print(exp_var_g)
    print(exp_var_b)
    
    # define target variance for each color channel [r,g,b]
    target_var = [0.8, 0.8, 0.8]
    com_target_idx = []
    for i in range(len(exp_var_list)):
        cum_var = np.cumsum(exp_var_list[i])
        # compare cumulative variance ratio with target and return index
        com_target_idx.append(np.argmax(cum_var>=target_var[i]))
    # note: this is the index of the max component
    # the actual number of components to keep is this plus one
    print(com_target_idx)
    #%%
    # calulate number of components with a explained variance ratio above the
    # mean explained variance ratio
    mean_exp_var = []
    num_com_above = []
    for i in range(len(exp_var_list)):
        # calculate mean
        mean_exp_var.append(np.mean(exp_var_list[i]))
        num_com_above.append(np.count_nonzero(exp_var_list[i] > mean_exp_var[i]))
    print(mean_exp_var)
    print(num_com_above)