from synthetic_real_dynamic_multicore import create_true, create_full_cadence, create_full_cadence_multicore, create_false, create_true_single_shot, create_true_faster
from skimage.transform import rescale, resize, downscale_local_mean
from numba import jit, prange, njit
import numpy as np 
import gc

@jit(nopython=True)
def pre_proc(data):
    data = np.log(data)
    data= data - data.min()
    data = data/data.max()
    return data

@jit(parallel=True)
def load_data_ED(data):
    print(data.shape)
    data_transform =  np.zeros((data.shape[0],6, 16,data.shape[3],1))
    for i in prange(data.shape[0]):
        data_transform[i,:,:,:,0]  = pre_proc(data[i,:,:,:] )
    return data_transform

def combine(data):
    new_data = np.zeros((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
    for i in prange(data.shape[0]):
        new_data[i*data.shape[1] : (i+1)*data.shape[1],:,:,:] = data[i,:,:,:,:]
    return new_data

def resize_par(data, factor):
    test =  np.zeros((data.shape[0], data.shape[1],data.shape[2],data.shape[3]//factor))
    print(data.shape, test.shape)
    for i in range(6):
        test[:,i,:,:] = downscale_local_mean(data[:,i,:,:], (1,1,factor))
    return test


def create_data_set_multicore( NUM_SAMPLES=10000, snr_base=20, snr_range = 10, factor=1, resize=8):

    print("Creating True")
    print(NUM_SAMPLES)
    data = create_full_cadence_multicore(create_true_faster, samples = NUM_SAMPLES,  snr_base=snr_base, snr_range=snr_range, factor =factor)
    print(data.shape)
    data = resize_par(data, factor=resize)
    data = combine(load_data_ED(data))
    print(data.shape)

    print("Creating False")
    false_data = abs(create_full_cadence_multicore(create_false,  samples = NUM_SAMPLES*6, snr_base=snr_base, snr_range=snr_range))
    false_data = resize_par(false_data, factor=resize)
    false_data = load_data_ED(false_data)

    print("Creating True")
    true_data_1 = create_full_cadence_multicore(create_true_faster,  samples = NUM_SAMPLES*3,  snr_base=snr_base, snr_range=snr_range, factor =factor)
    true_data_1 = resize_par(true_data_1, factor=resize)
    true_data_1 = load_data_ED(true_data_1)

    true_data_2 = create_full_cadence_multicore(create_true_single_shot, samples = NUM_SAMPLES*3,  snr_base=snr_base, snr_range=snr_range, factor =factor)
    true_data_2 = resize_par(true_data_2, factor=resize)
    true_data_2 = load_data_ED(true_data_2)

    true_data = np.concatenate((true_data_1,true_data_2),axis=0)
    print(true_data.shape)

    del true_data_1,true_data_2
    gc.collect()
    return data, false_data, true_data