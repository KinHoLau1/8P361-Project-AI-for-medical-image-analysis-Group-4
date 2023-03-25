# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:14:45 2023

@author: kinho
"""
from os.path import dirname, abspath
import os

# set working directory to location of py file
pypath = abspath(__file__)
dname = dirname(pypath)
os.chdir(dname)

import numpy as np

import glob
import pandas as pd
from matplotlib.pyplot import imread

from tensorflow.keras.models import model_from_json

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from PCA import IPCA_load, IPCA_reconstruction

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def predictions(files, pca_r, pca_g, pca_b, batch_size=1000, labels=False):
    '''
    Generates predictions using a trained model and returns dataframe containing
    image ids, predictions and , optionally, the true labels

    Parameters
    ----------
    files : list of image filepaths
    pca_r, pca_g, pca_b: IPCA objects
    batch_size : number of images to provide predictions for per batch.
    The default is 1000.
    labels : boolean which determines whether true labels are also added into
    the dataframe that is returned. Needed to generate ROC curve. 
    The default is False.

    Returns
    -------
    pred_pd : dataframe containg image ids and their predicted labels. Also
    contains true labels if labels is set to True.

    '''
    # prepare empty dataframe
    pred_pd = pd.DataFrame()
    
    # iterate over all iamges in dataset
    max_idx = len(files)
    for idx in range(0, max_idx, batch_size):
        # track progress
        if (idx+batch_size) >= max_idx:
            print('Indexes: %i - %i'%(idx, max_idx))
        else:
            print('Indexes: %i - %i'%(idx, idx+batch_size))
            
        # create dataframes each loop for temporary storage
        df = pd.DataFrame({'path': files[idx:idx+batch_size]})

        # get the image id 
        df['id'] = df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
        df['image'] = df['path'].map(imread)
        
        # collect images in array
        K_test = np.stack(df['image'].values)
        
        # apply the same preprocessing as during draining
        K_test = K_test.astype('float')/255.0
        K_test_pca = IPCA_reconstruction(K_test, pca_r, pca_g, pca_b)
        
        # generate predictions
        predictions = model.predict(K_test_pca)
        
        df['prediction'] = predictions
        # append data to final dataframe
        if labels == True:
            df['label'] = [int(file.partition('valid/')[2][0]) for file in files[idx:idx+batch_size]]
            pred_pd = pd.concat([pred_pd, df[['id', 'prediction','label']]])
        else:
            pred_pd = pd.concat([pred_pd, df[['id', 'prediction']]])
        
    return pred_pd

#%%
#Change these variables to point at the locations and names of the test dataset and your models.
test_path = 'C:/8P361/test'
val_path = 'C:/8P361/valid'

# load IPCA models
#ret_var = '90'
#ret_var = '80'
#ret_var = '70'
ret_var = '60'

pca_r,pca_g,pca_b = IPCA_load(ret_var)

# load CNN model
parent = dirname(dirname(abspath(__file__)))
model_folder = parent + '\CNN Models\\'
model_filepath = model_folder + 'IPCA_'+ret_var+'_model' + '.json' 
model_weights_filepath = model_folder + 'IPCA_'+ret_var+'_model' + '_weights.hdf5'

# load model and model weights
json_file = open(model_filepath, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(model_weights_filepath)

# open the test and validation sets in batches and make predictions
test_files = glob.glob(test_path + '/*.tif') 
val_files = glob.glob(val_path + '/0/*.jpg') + glob.glob(val_path + '/1/*.jpg') 

# perform predictions
pred_test = predictions(test_files, pca_r, pca_g, pca_b,1000)
pred_val = predictions(val_files, pca_r, pca_g, pca_b, 1000, True)
#%%
# ROC analysis
fpr, tpr, thresholds = roc_curve(pred_val['label'], pred_val['prediction'])
auc_model = auc(fpr, tpr)
#%%
# visualise ROC curve
plt.plot([0,100],[0,100],'r--', label='Random classifier')
plt.plot(fpr*100, tpr*100, label='Convolutional Neural Network')
plt.xlabel('False positive rate (1-specificity) [%]')
plt.ylabel('True positive rate (sensitivity) [%]')
plt.title('ROC curve for '+ret_var+'% retained variance')
plt.xlim(0,100)
plt.ylim(0,100)
plt.legend()
plt.show()
print(auc_model)
#%%
# rename prediction column
pred_test.rename(columns={'prediction':'label'},inplace=True)
# check first columns
pred_test.head()
# save submission
sub_folder = parent+'\Submissions\\'
pred_test.to_csv(sub_folder+'submission_'+ret_var+'.csv', index = False, header = True)