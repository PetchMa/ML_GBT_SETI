import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange, njit
from blimpy import Waterfall
import time
import random

# data preprocessing operations 
# Goal is to take a full cadence and shape it into something usable 
# for a wide range of ML pipelines
class preprocess(object):
    def __init__(self, cadence, start, end):
        self.A1_name = cadence[0]
        self.A2_name = cadence[2]
        self.A3_name = cadence[4]

        self.B_name = cadence[1]
        self.C_name = cadence[3]
        self.D_name = cadence[5]

        self.start = start
        self.end = end
    
    # We get the data for a strict shape of freq 256, and time 16 and we stack them together. 
    # returns the stack of all the slices in order and log normalized and scaled between 1 and 0.
    def get_data(self):
        print("Getting Data")
        Waterfall(self.A1_name, load_data=False).info()

        A1 = Waterfall(self.A1_name, f_start=self.start, f_stop=self.end, max_load=10).data
        B = Waterfall(self.B_name, f_start=self.start, f_stop=self.end, max_load=10).data
        A2 = Waterfall(self.A2_name, f_start=self.start, f_stop=self.end, max_load=10).data
        C = Waterfall(self.C_name, f_start=self.start, f_stop=self.end, max_load=10).data
        A3 = Waterfall(self.A3_name, f_start=self.start, f_stop=self.end, max_load=10).data
        D = Waterfall(self.D_name, f_start=self.start, f_stop=self.end, max_load=10).data

        start_pre = time.time()
        A1 =self.shaping_data(A1)
        B =self.shaping_data(B)
        A2 =self.shaping_data(A2)
        C =self.shaping_data(C)
        A3 =self.shaping_data(A3)
        D =self.shaping_data(D)

        data = self.combine_cadence(A1,A2,A3,B,C,D)

        print("Execution Time: "+str(time.time()-start_pre))
        return data
    
    # shaping the data by stacking them together. 
    @jit(parallel=True)
    def shaping_data(self, data):
        samples = data.shape[2]//256
        new_data = np.zeros((samples, 16, 256, 1))
        for i in prange(samples):
            new_data[i,:,:,0] = data[:,0,i*256:(i+1)*256]
        return new_data

    # preprocess the data with the following operations acclerated via numba
    @jit(nopython=True, parallel=True)
    def pre_proc(self, data):
    #   data= data - data.min()+1
        data = np.log(data)
        data= data - data.min()
        data = data/data.max()
        return data

    #combing all the data together 
    @jit(parallel=True, nopython=True)
    def combine_cadence(A1,A2,A3,B,C,D):
        samples = A1.shape[0]
        data = np.zeros((samples,6, 16, 256, 1))
        for i in prange(samples):
            data[i,0,:,:,:] = A1[i,:,:,:]
            data[i,1,:,:,:] = B[i,:,:,:]
            data[i,2,:,:,:] = A2[i,:,:,:]
            data[i,3,:,:,:] = C[i,:,:,:]
            data[i,4,:,:,:] = A3[i,:,:,:]
            data[i,5,:,:,:] = D[i,:,:,:]
            data[i,:,:,:,:] = self.pre_proc(data[i,:,:,:,:] )
        return data


