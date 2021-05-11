import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, njit
from blimpy import Waterfall
import time
import random

# data preprocessing operations 
# Goal is to take a full cadence and shape it into something usable 
# for a wide range of ML pipelines

# We get the data for a strict shape of freq 256, and time 16 and we stack them together. 
# returns the stack of all the slices in order and log normalized and scaled between 1 and 0.
def get_data(cadence, start, end):
    print("Getting Data")
    Waterfall(cadence[0], load_data=False).info()

    A1 = Waterfall(cadence[0], f_start=start, f_stop=end, max_load=10).data
    B = Waterfall(cadence[1], f_start=start, f_stop=end, max_load=10).data
    A2 = Waterfall(cadence[2], f_start=start, f_stop=end, max_load=10).data
    C = Waterfall(cadence[3], f_start=start, f_stop=end, max_load=10).data
    A3 = Waterfall(cadence[4], f_start=start, f_stop=end, max_load=10).data
    D = Waterfall(cadence[5], f_start=start, f_stop=end, max_load=10).data

    start_pre = time.time()
    A1 =shaping_data(A1)
    B =shaping_data(B)
    A2 =shaping_data(A2)
    C =shaping_data(C)
    A3 =shaping_data(A3)
    D =shaping_data(D)
    D = pre_proc(D)
    # data = self.combine_cadence(A1,A2,A3,B,C,D)

    print("Execution Time: "+str(time.time()-start_pre))
    return D

# shaping the data by stacking them together. 
@jit(parallel=True)
def shaping_data( data):
    samples = data.shape[2]//256
    new_data = np.zeros((samples, 16, 256, 1))
    for i in prange(samples):
        new_data[i,:,:,0] = data[:,0,i*256:(i+1)*256]
    return new_data

# preprocess the data with the following operations acclerated via numba
@njit(nopython=True, parallel=True)
def pre_proc( data):
#   data= data - data.min()+1
    data = np.log(data)
    data = data/data.max()
    data= data - data.min()
    return data

#combing all the data together 
@jit(parallel=True, nopython=True)
def combine_cadence( A1,A2,A3,B,C,D):
    samples = A1.shape[0]
    data = np.zeros((samples,6, 16, 256, 1))
    for i in prange(samples):
        data[i,0,:,:,:] = A1[i,:,:,:]
        data[i,1,:,:,:] = B[i,:,:,:]
        data[i,2,:,:,:] = A2[i,:,:,:]
        data[i,3,:,:,:] = C[i,:,:,:]
        data[i,4,:,:,:] = A3[i,:,:,:]
        data[i,5,:,:,:] = D[i,:,:,:]
        data[i,:,:,:,:] = pre_proc(data[i,:,:,:,:] )
    return data


