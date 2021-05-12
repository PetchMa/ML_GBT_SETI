import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
import tensorflow as tf

def weak_cadence_pattern(labels):
    return labels[0]!=labels[1] or labels[1]!=labels[2] and labels[2]!= labels[3] or labels[3]!=labels[4] and labels[4]!=labels[5] 

def strong_cadence_pattern(labels):
    return labels[0]!=labels[1] and labels[1]!=labels[2] and labels[2]!= labels[3] and labels[3]!=labels[4] and labels[4]!=labels[5] 

@jit(parallel=True)
def combine(data):
    new_data = np.zeros((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
    for i in prange(data.shape[0]):
        new_data[i*data.shape[1] : (i+1)*data.shape[1],:,:,:] = data[i,:,:,:,:]
    return new_data

def sample_creation(inputs):
    z_mean = inputs[0]
    z_log_var = inputs[1]
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def classification_data(target_name,cadence, model, out_dir, iterations=6):
    f_hit_start = []
    f_hit_end = []
    header = Waterfall(cadence[0]).header
    end = header['fch1']
    start = header['fch1']+ header['nchans']*header['foff']
    interval = (end-start)/iterations
    WINDOW_SIZE = abs(256*header['foff'])
    print(end)
    freq_ranges = []
    for i in range(iterations):
        f_start = start+i *interval
        f_stop = start+(i+1)*interval
        freq_ranges.append([f_start, f_stop])
    print(freq_ranges)
    # #execution looop:
    # for index in range(iterations):
    #     data = get_data(cadence,start =freq_ranges[i][0],end =freq_ranges[i][1])
    #     num_samples = data.shape[0]
    #     cadence_length = data.shape[1]
    #     for n in range(num_samples):
    #         hit_start = freq_ranges[index][0] + n*WINDOW_SIZE
    #         hit_end = hit_start + WINDOW_SIZE
    #         print(hit_start)
    #         f_hit_start.append(hit_start)
    #         f_hit_end.append(hit_end)

    # candidates = {'f_start':f_hit_start,'f_end':f_hit_start }
    # df = pd.from_dict(candidates)
    # df.to_csv(out_dir+"/"+target_name+".csv")
    # print(len(f_hit_start))
