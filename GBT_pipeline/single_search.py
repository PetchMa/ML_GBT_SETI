import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
sys.path.insert(1, '../ML_Training')
from execute_model import model_predict_distribute
from preprocess import pre_proc
from numba import jit, prange, njit
from blimpy import Waterfall
import time
import random
from sklearn.cluster import SpectralClustering
from pandas import DataFrame as pd
import tensorflow as tf
from multiprocessing import Pool
import functools
import warnings
import logging
from numpy.linalg import norm
from sklearn.metrics import silhouette_score
import warnings

tf.get_logger().setLevel('INFO')

def strong_cadence_pattern(labels):
    return labels[0]!=labels[1] and labels[1]!=labels[2] and labels[2]!= labels[3] and labels[3]!=labels[4] and labels[4]!=labels[5] 

def screening(data, labels, SNR):
    fit = silhouette_score(data,labels)
    if fit > 0.0 and SNR>0:
        return True
    return False

def compute_parallel(result, flag,SNR, n):
    warnings.filterwarnings("ignore")
    labels = result[n*6: (n+1)*6, : ]
    labels = SpectralClustering(n_clusters=2, assign_labels="discretize", 
                random_state=0).fit_predict(labels)
    if strong_cadence_pattern(labels)==True:
        if screening(result[n*6: (n+1)*6, : ], labels, SNR[n]):
            return True
        else:
            return False
    else:
        return False

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


@jit(parallel=True)
def check(data, flag):
    correct= 0
    for i in prange(len(data)):
        if data[i] == flag:
            correct+=1
    return correct

def search(data, model, flag):
    SNR = []
    for i in range(data.shape[0]):
        SNR.append(data[i].max()/np.mean(data[i]))
    print(data.shape)
    for i in range(data.shape[0]):
        data[i,:,:,:] = pre_proc(data[i,:,:,:] )
    num_samples = data.shape[0]
    cadence_length = data.shape[1]
    data = data[..., np.newaxis]
    
    print("Collapse Data")
    data = combine(data)
    print("Push Through Neural Net")
    net = time.time()
    result = model.predict(data, batch_size=5000, use_multiprocessing =True)[2]
    print("Parallel Spectral Clustering")
    cluster = time.time()
    with Pool(39) as p:
        result = p.map(functools.partial(compute_parallel,result, flag, SNR), range(num_samples))
    print(check(result, flag)/len(result))
    return check(result, flag)/len(result)


def search_model_eval(data, flag):
    with Pool(39) as p:
        result = p.map(functools.partial(compute_parallel,data,SNR, flag), range(data.shape[0]//6))
    return check(result)/len(result)