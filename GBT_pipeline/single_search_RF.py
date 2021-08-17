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

def strong_cadence_pattern(label_1, label_2, thresh=0.5):
    if label_2 > thresh:
        return True
    else:
        return False
def check_data(label,thresh=0.5):
    check = []
    for i in range(label.shape[0]):
        check.append(strong_cadence_pattern(label[i,0], label[i,1], thresh))
    return check

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

def recombine(data):
    result = []
    for k in range(data.shape[0]//6):
        result.append(data[k*6:(k+1)*6,:].ravel())
    result = np.array(result)
    return result

def search(data, model, forest,  flag, thresh=0.5):
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
    print("Parallel Random Forest Clustering")
    cluster = time.time()
    result = recombine(result)
    result = forest.predict_proba(result)
    if flag:
        mean = np.mean(result[:,1])
    else:
        mean = np.mean(result[:,0])
    labels = check_data(result, thresh)
    print(check(labels, flag)/len(labels), mean)
    return check(labels, flag)/len(labels), mean
