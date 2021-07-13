import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, njit
from blimpy import Waterfall
import time
import random
import warnings 
from tqdm  import tqdm
import sys
import os, psutil
import gc
import matplotlib.pyplot as plt 
from skimage.transform import rescale, resize, downscale_local_mean
# data preprocessing operations 
# Goal is to take a full cadence and shape it into something usable 
# for a wide range of ML pipelines
WIDTH_BIN = 2048
FACTOR = 4

# We get the data for a strict shape of freq 256, and time 16 and we stack them together. 
# returns the stack of all the slices in order and log normalized and scaled between 1 and 0.
def get_data(cadence, start, end):
    # Waterfall(cadence[0], load_data=False).info()
    process = psutil.Process(os.getpid())

    start_pre = time.time()


    A1 =shaping_data(Waterfall(cadence[0], f_start=start, f_stop=end).data)
    A1 = downscale_local_mean(A1, (1, 1,FACTOR, 1,))
    num_samples = A1.shape[2]//(2048//FACTOR)
    snr = []
    # temp = np.sum(A1, axis=0)
    for i in range(num_samples):
          snr.append(np.amax(A1[:,:,i*256:(i+1)*256])/np.mean(A1[:,:,i*256:(i+1)*256]))

    B =shaping_data( Waterfall(cadence[1], f_start=start, f_stop=end).data)
    B = downscale_local_mean(B, (1, 1,FACTOR, 1,))
    A2 =shaping_data( Waterfall(cadence[2], f_start=start, f_stop=end).data)
    A2 = downscale_local_mean(A2, (1, 1,FACTOR, 1,))
    C = shaping_data(Waterfall(cadence[3], f_start=start, f_stop=end).data)
    C = downscale_local_mean(C, (1, 1,FACTOR, 1,))
    A3 = shaping_data(Waterfall(cadence[4], f_start=start, f_stop=end).data)
    A3 = downscale_local_mean(A3, (1, 1,FACTOR, 1,))
    D = shaping_data(Waterfall(cadence[5], f_start=start, f_stop=end).data)
    D = downscale_local_mean(D, (1, 1,FACTOR, 1,))


    # plt.figure(figsize=(10,4))
    # plt.xlabel("Fchans")
    # plt.ylabel("Time")
    # plt.imshow(A1[10,:,:,0], interpolation='nearest', cmap=plt.get_cmap('hot'))
    # plt.savefig('test.png')
   
    data = combine_cadence(A1,A2,A3,B,C,D)
    del A1, A2, A3, B, C , D
    gc.collect()
    print("Data Load Execution Time: "+str(time.time()-start_pre))
    # print(data.shape)
    return data, snr

# shaping the data by stacking them together. 
@jit(parallel=True)
def shaping_data(data):
    samples = data.shape[2]//WIDTH_BIN
    new_data = np.zeros((samples, 16, WIDTH_BIN, 1))
    for i in prange(samples):
        new_data[i,:,:,0] = data[:,0,i*WIDTH_BIN:(i+1)*WIDTH_BIN]
    return new_data

# preprocess the data with the following operations acclerated via numba
@njit(nopython=True)
def pre_proc(data):
#   data= data - data.min()+1
    data = np.log(data)
    data= data - data.min()
    data = data/data.max()
    return data

#combing all the data together 
@jit(parallel=True)
def combine_cadence(A1,A2,A3,B,C,D):
    samples = A1.shape[0]
    data = np.zeros((samples,6, 16, WIDTH_BIN//FACTOR, 1))
    for i in prange(samples):
        data[i,0,:,:,:] = A1[i,:,:,:]
        data[i,1,:,:,:] = B[i,:,:,:]
        data[i,2,:,:,:] = A2[i,:,:,:]
        data[i,3,:,:,:] = C[i,:,:,:]
        data[i,4,:,:,:] = A3[i,:,:,:]
        data[i,5,:,:,:] = D[i,:,:,:]
        data[i,:,:,:,:] = pre_proc(data[i,:,:,:,:] )
    
    return data

def resize_par(data, factor):
    test =  np.zeros((data.shape[0], data.shape[1],data.shape[2],data.shape[3]//factor))
    print(data.shape, test.shape)
    for i in range(6):
        test[:,i,:,:] = downscale_local_mean(data[:,i,:,:], (1,1,factor))
    return test

