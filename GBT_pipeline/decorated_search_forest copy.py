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
from sklearn.tree import DecisionTreeClassifier
import joblib
import matplotlib.pyplot as plt
from random import random

METRIC = [0.7,0.8,0.9,0.8,
        0.7,0.7,0.7,0.8,
        0.95,0.95,0.9,0.8,
        0.7,0.7,0.8,0.95]
WIDTH_BIN = 4096

def batch_tree(forest, result, load, n):
    forest = forest.set_params(n_jobs = 1)
    start = load[n][0]
    end = load[n][len(load[n])-1]
    probability = forest.predict_proba(result[start:end])
    return probability

def unravel(data):
    frame = np.zeros((data.shape[0]//6, data.shape[1]*6))
    for i in range(data.shape[0]//6):
        frame[i,:] = data[i*6:(i+1)*6,:].ravel()
    return frame


# Strongest cadence pattern where only on,off,on,off,on,off patterns are accepeted. 
def strong_cadence_pattern(labels, WINDOW_SIZE, index, freq_ranges ):
    candidates = []
    for i in range(labels.shape[0]):
        if labels[i][1]>labels[i][0]:
            hit_start = freq_ranges[index][1] - (i+1)*WINDOW_SIZE
            hit_end = freq_ranges[index][1] - (i)*WINDOW_SIZE
            candidates.append([hit_start, hit_end, labels[i][1], i*6])
    return candidates
        

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
def classification_data(target_name,cadence, model,forest, out_dir, iterations=6, name_index=0):
    process = psutil.Process(os.getpid())
    # Create empty list to store the results
    f_hit_start = []
    f_hit_end = []
    # Get the header information
    header = Waterfall(cadence[0], load_data = False).header

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
        data = combine(data)
        # Feed through neural network
        net = time.time()
        result = model.predict(data, batch_size=10000)[2]
        num_samples = result.shape[0]//6
        cadence_length = 6
        print("Push Through Neural Net: "+str(time.time()-net))
        print(process.memory_info().rss*1e-9) 
        print(result.shape)
        cluster = time.time()
        result = unravel(result)
        print("check nan here")
        if np.isfinite(result).all()==False:
            print("NAN ERROR")
            ar_inf = np.where(np.isinf(result))
            result = np.nan_to_num(result, posinf=100, neginf=-100)
            print(np.isnan(np.sum(result)))
            f = open("result/log.txt", "a")
            f.write("Nan found here:"+target_name+"_"+str(freq_ranges[index])+" "+str(ar_inf)+'\n')
            f.close()
        # Run forest in parallel with one idle core
        
        
        #DIVIDE WORKLOAD 
        
        probability = forest.predict_proba(result)
        # CORES =20
        # load_ = np.arange(num_samples)
        # load = list(np.array_split(load_, CORES))
        # for i in range(CORES):
        #     load[i] =load[i].tolist()
        # print("check")
        # with Pool(CORES) as p:
        #     probability = p.map(functools.partial(batch_tree, forest, result, load),range(CORES))
        print("Random forest "+str(time.time()-cluster))
        # Shows the results
        flat_list = strong_cadence_pattern(probability, WINDOW_SIZE, index, freq_ranges )
        print("Number of Candidates found: "+str(len(flat_list)))
        all_candidates.append(flat_list)

        key = int(len(flat_list)*random())
        random_index = flat_list[key][3]
        plt.figure(figsize=(10,4))
        plt.xlabel("Fchans")
        plt.ylabel("Time")
        print(random_index)
        plt.title(str(flat_list[key][0])
                +" _" +str(flat_list[key][1])
                +" _" +str(flat_list[key][3]))
        plt.imshow(data[random_index,:,:,0], interpolation='nearest', cmap=plt.get_cmap('hot'))
        plt.savefig(target_name+"_"+str(flat_list[key][0])+".png")
        print(data.shape)

        del result, data
        gc.collect()

    final_set = []
    for k in range(len(all_candidates)):
        for el in all_candidates[k]:
            final_set.append(el)
    print("Number of Final Candidates "+str(len(final_set)))
    df = pd.DataFrame(final_set, columns =['start_freq', 'end_freq', 'probability', 'index'], dtype = float)
    df.to_csv(out_dir+target_name+"_"+str(name_index)+".csv")
    return final_set
    



