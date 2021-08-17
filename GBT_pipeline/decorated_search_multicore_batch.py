# ============================================================
# Author: Peter Xiangyuan Ma
# Date: May 19 2021
# Purpose: split the search functionality into smaller chuncks 
# to be called by the full_search.py pipeline. This code, loops 
# through chunks of the cadence and preprocesses it, 
# feed into neural network and then runs the clustering algorithm
# in parallel using multiple CPU cores. 
# ============================================================

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
sys.path.insert(1, '../ML_Training')
from execute_model import model_predict_distribute
from preprocess_dynamic import get_data
from numba import jit, prange, njit
from blimpy import Waterfall
import time
import random
from sklearn.cluster import SpectralClustering
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
import functools
import warnings
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import os, psutil
import warnings 
import gc

METRIC = [0.7,0.8,0.9,0.8,
        0.7,0.7,0.7,0.8,
        0.95,0.95,0.9,0.8,
        0.7,0.7,0.8,0.95]
WIDTH_BIN = 4096
def screening(data, labels, snr, index):
    fit = silhouette_score(data,labels)
    if fit > METRIC[index]:
        return True
    else:
        return False
def extra_screening(data, labels, snr, index):
    fit = silhouette_score(data,labels)
    if fit > METRIC[index]:
        return True, fit
    else:
        return False, fit

def batch_compute_parallel(result, cadence_length,WINDOW_SIZE,index,freq_ranges, snr, load,n):
    solutions = []
    for el in load[n]:
        solutions.append(compute_parallel(result, cadence_length,WINDOW_SIZE,index,freq_ranges, snr, el))
    solutions = [i for i in solutions if i]
    return solutions


# Function takes in small distributed chunks of data and runs spectral clustering on the data set
# returns a list of candidates with the frequency range. 
def compute_parallel(result, cadence_length,WINDOW_SIZE,index,freq_ranges, snr, n):
    warnings.filterwarnings("ignore")
    # spectral clustering
    if snr[n] > 1.7:
        labels = SpectralClustering(n_clusters=2, assign_labels="discretize", 
                    random_state=0).fit_predict( result[n*cadence_length: (n+1)*cadence_length, : ])
        if strong_cadence_pattern(labels):
            if screening(result[n*6: (n+1)*6, : ], labels,snr[n], index):
                screen_flag, fit = extra_screening(result[n*6: (n+1)*6, : ], labels,snr[n], index)
                # Windowsize is the width of the snipet in terms of Hz
                hit_start = freq_ranges[index][1] - (n+1)*WINDOW_SIZE
                hit_end = freq_ranges[index][1] - (n)*WINDOW_SIZE
                # Computes the frequency start and end of this given window
                return [hit_start, hit_end, fit, snr[n]]


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
    process = psutil.Process(os.getpid())
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
    WINDOW_SIZE = abs(WIDTH_BIN*header['foff'])
    # Break down the frequency into chuncks of smaller sizes to processes
    freq_ranges = []
    for i in range(iterations):
        f_start = start+i *interval
        f_stop = start+(i+1)*(interval)
        freq_ranges.append([f_start, f_stop])
    print(freq_ranges)
    all_candidates = []
    #execution looop through each of the individual chunks of data
    for index in range(iterations):
        print(target_name+ " Iteration: "+str(index)+ " Range: "+str(freq_ranges[index]))
        print(process.memory_info().rss*1e-9)     
        # Collapse the data without the cadence axis, however keeping the order of the cadences 
        data, snr = get_data(cadence, start =freq_ranges[index][0], end =freq_ranges[index][1])
        data = resize_par(data, factor=4)
        data = combine(data)
        # Feed through neural network
        net = time.time()
        result = model.predict(data, batch_size=10000)[2]
        num_samples = result.shape[0]//6
        cadence_length = 6
        print("Push Through Neural Net: "+str(time.time()-net))
        print(process.memory_info().rss*1e-9) 
        # Run spectral clustering in parallel with one idle core
        CORES =39
        load_ = np.arange(num_samples)
        load = list(np.array_split(load_, CORES))
        for i in range(CORES):
            load[i] =load[i].tolist()
        cluster = time.time()
        with Pool(CORES) as p:
            candidates = p.map(functools.partial(batch_compute_parallel, result, cadence_length,WINDOW_SIZE,index, freq_ranges, snr, load), 
                                range(CORES))
        print("Parallel Spectral Clustering: "+str(time.time()-cluster))
        # Shows the results
        flat_list = [item for sublist in candidates for item in sublist]
        print("Number of Candidates found: "+str(len(flat_list)))
        all_candidates.append(flat_list)
        del result
        del candidates
        del data
        gc.collect()
    final_set = []
    for k in range(len(all_candidates)):
        for el in all_candidates[k]:
            final_set.append(el)
    print("Number of Final Candidates "+str(len(final_set)))
    df = pd.DataFrame(final_set, columns =['start_freq', 'end_freq', 'fit','SNR'], dtype = float)
    df.to_csv(out_dir+target_name+".csv")
    return final_set
    



