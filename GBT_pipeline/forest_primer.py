import sys
sys.path.insert(1, '../ML_Training')
sys.path.insert(2, '../GBT_pipeline')
sys.path.insert(3, '../test_bench')
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from numba import jit, prange, njit
from blimpy import Waterfall
import time
import random
from synthetic_real_dynamic import create_true, create_full_cadence, create_false, create_true_single_shot, create_true_faster
import math
from sklearn.metrics import silhouette_score


from preprocess_dynamic import get_data
from single_search import search_model_eval, combine
from skimage.transform import rescale, resize, downscale_local_mean
import gc
from data_generation import create_data_set
from sklearn.ensemble import RandomForestClassifier
import os
import joblib


def combine(data):
    new_data = np.zeros((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
    for i in prange(data.shape[0]):
        new_data[i*data.shape[1] : (i+1)*data.shape[1],:,:,:] = data[i,:,:,:,:]
    return new_data

def model_compute(data, model):
    print("combine")
    data = combine(data)
    result= model.predict(data, batch_size=500)[2]
    print("recombine")
    return result

def recombine(data):
    result = []
    for k in range(data.shape[0]//6):
        result.append(data[k*6:(k+1)*6,:].ravel())
    result = np.array(result)
    return result


def forest_primer(model, plate_train):
    print("preparing Forest Model")
    # Forest generation

    print("Create train")
    data, false_data_train, true_data_train = create_data_set(plate_train, 
    NUM_SAMPLES=4000, snr_base=10, snr_range = 50, factor=1)
    del plate_train, data
    gc.collect()

    true_train = model_compute(true_data_train, model)
    false_train =model_compute(false_data_train, model)

    true_train = recombine(true_train)
    false_train = recombine(false_train)

    train = np.concatenate((true_train,false_train))
    print(train.shape)
    true_labels = np.ones((true_train.shape[0]))
    true_labels[:]=1

    false_labels = np.zeros((false_train.shape[0]))
    false_labels[:]=0
    labels = np.concatenate((true_labels,false_labels))
    print(labels.shape)
    train, labels = shuffle(train, labels)
    # Create the model with 100 trees
    print("train")
    tree = RandomForestClassifier(n_estimators=1000,     
                               bootstrap = True,
                               max_features = 'sqrt',n_jobs=-1)
    tree.fit(train, labels)
    # joblib.dump(tree, "../test_bench/random_forest_10000.joblib")
    return tree 





