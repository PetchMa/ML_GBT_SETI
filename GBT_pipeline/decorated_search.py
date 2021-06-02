# ============================================================
# Author: Peter Xiangyuan Ma
# Date: May 19 2021
# Purpose: split the search functionality into smaller chuncks 
# to be called by the full_search.py pipeline. This code, loops 
# through chunks of the cadence and preprocesses it, 
# feed into neural network and then runs the clustering algorithm
# using SINGLE CPU core. 
# ============================================================

import numpy as np
import sys
sys.path.insert(1, '../ML_Training')
from execute_model import model_predict_distribute
from preprocess import get_data

from numba import jit, prange, njit
from blimpy import Waterfall
import time
import random
from sklearn.cluster import SpectralClustering
from pandas import DataFrame as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# Weakest cadence pattern where anything with a on, and adjacent off pattern is accepted
def weak_cadence_pattern(labels):
    return labels[0]!=labels[1] or labels[1]!=labels[2] and labels[2]!= labels[3] or labels[3]!=labels[4] and labels[4]!=labels[5] 

# Strongest cadence pattern where only on,off,on,off,on,off patterns are accepeted. 
def strong_cadence_pattern(labels):
    return labels[0]!=labels[1] and labels[1]!=labels[2] and labels[2]!= labels[3] and labels[3]!=labels[4] and labels[4]!=labels[5] 

# Combines all the data together into one chunkc of data instead of in separate cadence samples. 
@jit(parallel=True)
def combine(data):
    new_data = np.zeros((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
    for i in prange(data.shape[0]):
        # Takes set of cadences and collapsing it down without that cadence axis, order is preserved. 
        new_data[i*data.shape[1] : (i+1)*data.shape[1],:,:,:] = data[i,:,:,:,:]
    return new_data


# computes the statistical sampling from the two layers of mean and variance
def sample_creation(inputs):
    z_mean = inputs[0]
    z_log_var = inputs[1]
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Classification function
def classification_data(target_name,cadence, model, out_dir, iterations=6):
    # Create empty list to store the results
    f_hit_start = []
    f_hit_end = []
    # Get the header information
    header = Waterfall(cadence[0]).header
    # Get the maximum freq in MHz
    end = header['fch1']
    # calculate the start by taking the resolution time thes number of samples and then adding it to the maximum [it is negative resolution]
    start = header['fch1']+ header['nchans']*header['foff']
    interval = (end-start)/iterations
    # Compute the window size in MHz
    WINDOW_SIZE = abs(256*header['foff'])
    # Break down the frequency into chuncks of smaller sizes to processes
    freq_ranges = []
    for i in range(iterations):
        f_start = start+i *interval
        f_stop = start+(i+1)*(interval)
        freq_ranges.append([f_start, f_stop])
    
    #execution looop:
    for index in range(iterations):
        data = get_data(cadence,start =freq_ranges[index][0],end =freq_ranges[index][1])
        num_samples = data.shape[0]
        cadence_length = data.shape[1]
        data = combine(data)
        result = model.predict(data, batch_size=5000, use_multiprocessing =True)
        print(result[0].shape)
        result =  sample_creation(result).numpy()
        for n in range(num_samples):
            labels = result[n*cadence_length: (n+1)*cadence_length, : ]
            labels = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit_predict(labels)
            if True:
                hit_start = freq_ranges[index][0] + n*WINDOW_SIZE
                hit_end = hit_start + WINDOW_SIZE
                f_hit_start.append(hit_start)
                f_hit_end.append(hit_end)

    candidates = {'f_start':f_hit_start,'f_end':f_hit_start }
    df = pd.from_dict(candidates)
    df.to_csv(out_dir+"/"+target_name+".csv")
    print(len(f_hit_start))
