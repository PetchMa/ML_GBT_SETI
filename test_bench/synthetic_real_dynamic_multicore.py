import numpy as np
from blimpy import Waterfall 
from astropy import units as u
import setigen as stg
import matplotlib.pyplot as plt
from random import random
from numba import jit, prange, njit
import math
from multiprocessing import Pool
import functools
from sys import getsizeof

WIDTH_BIN = 2048

def new_frame(mean = 58348559, snr_power = 1):
    snr=(random()*10+200)
    drift=(random()*2+1)*(-1)**(int(random()*3+1))
    start = int(random()*126)+100
    frame = stg.Frame(fchans=WIDTH_BIN*u.pixel,
                    tchans=16*u.pixel,
                    df=2.7939677238464355*u.Hz,
                    dt=18.25361108*u.s,
                    fch1=6095.214842353016*u.MHz)
    noise = frame.add_noise(x_mean=mean, noise_type='chi2')
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=start),
                                              drift_rate=drift*u.Hz/u.s),
                            stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                            stg.gaussian_f_profile(width=50*u.Hz),
                            stg.constant_bp_profile(level=1))
    return frame.data

def intersection(m1,m2,b1,b2):
    solution = (b2-b1)/(m1-m2)
    y = m1*solution+b1
    if y>=80 and y<=96:
        return False
    elif y>=64 and y<=80:
        return False
    elif y>=32 and y<=48:
        return False
    elif y>=0 and y<=16:
        return False
    else:
        return True

def new_cadence(data, snr):
    witdh = random()*25+5
    start = int(random()*WIDTH_BIN//2)+WIDTH_BIN//2
    if (-1)**(int(random()*3+1)) > 0:
        true_slope = (96/start)
        slope = (true_slope)*(18.25361108/2.7939677238464355)+random()*1e-12
    else:
        true_slope = (96/(start-WIDTH_BIN))
        slope = (true_slope)*(18.25361108/2.7939677238464355)-random()*1e-12
    drift = -1*(1/slope)
    b = 96-true_slope*(start)
    frame = stg.Frame.from_data(df=2.7939677238464355*u.Hz,
                            dt=18.25361108*u.s,
                            fch1=0*u.MHz,
                            data=data)
    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=start),
                                              drift_rate=drift*u.Hz/u.s),
                            stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                            stg.gaussian_f_profile(width=witdh*u.Hz),
                            stg.constant_bp_profile(level=1))
    return frame.data, true_slope, b

    
def create_true(plate,  snr_base=300, snr_range=10, factor=1, index=1):
    
    index = int(plate.shape[0]*random())
    total = np.zeros((6,plate.shape[2],plate.shape[3]  ))
    base = plate[index,:,:,:]
    data = np.zeros((96,WIDTH_BIN))
    for el in range(6):
        data[16*el: (el+1)*16,:] = base[el,:,:]
    
    while True:
        snr=(random()*snr_range+snr_base)
        cadence, m1,b1 =  new_cadence(data, snr)
        injection_plate, m2, b2=  new_cadence(cadence, snr*factor)
        if m1!=m2:
            if intersection(m1,m2,b1,b2):
                break
    total[0,:,:] = injection_plate[0:16,:]
    total[1,:,:] = cadence[16:32,:]
    total[2,:,:] = injection_plate[32:48,:]
    total[3,:,:] = cadence[48:64,:]
    total[4,:,:] = injection_plate[64:80,:]
    total[5,:,:] = cadence[80:96,:]
    return total

def create_true_faster(plate,  snr_base=300, snr_range=10, factor=1, index=1):
    # print( snr_base, snr_range, factor, index)
    index = int(plate.shape[0]*random())
    total = np.zeros((6,plate.shape[2],plate.shape[3] ))
    base = plate[index,:,:,:]
    data = np.zeros((96,WIDTH_BIN))
    for el in range(6):
        data[16*el: (el+1)*16,:] = base[el,:,:]
    snr=(random()*snr_range+snr_base)
    cadence, m1,b1 =  new_cadence(data, snr)
       
    injection_plate, m2, b2=  new_cadence(cadence, snr*factor)

    total[0,:,:] = injection_plate[0:16,:]
    total[1,:,:] = cadence[16:32,:]
    total[2,:,:] = injection_plate[32:48,:]
    total[3,:,:] = cadence[48:64,:]
    total[4,:,:] = injection_plate[64:80,:]
    total[5,:,:] = cadence[80:96,:]

    # print(getsizeof(total))
    return total

def create_true_single_shot(plate, snr_base=10, snr_range=5, factor = 1, index=1):
    index = int(plate.shape[0]*random())
    total = np.zeros((6,plate.shape[2],plate.shape[3]  ))
    base = plate[index,:,:,:]
    data = np.zeros((96,WIDTH_BIN))
    for el in range(6):
        data[16*el: (el+1)*16,:] = base[el,:,:]
    
    snr=(random()*snr_range+snr_base)
    injection_plate, m2, b2=  new_cadence(data, snr)
    total[0,:,:] = injection_plate[0:16,:]
    total[1,:,:] = data[16:32,:]
    total[2,:,:] = injection_plate[32:48,:]
    total[3,:,:] = data[48:64,:]
    total[4,:,:] = injection_plate[64:80,:]
    total[5,:,:] = data[80:96,:]
    return total

# @jit(nopython=True)
def create_false(plate, snr_base=300, snr_range=10, factor = 1, index=1):
    choice = random()
    if choice > 0.5:
        index = int(plate.shape[0]*random())
        total = np.zeros((6,plate.shape[2],plate.shape[3] ))
        base = plate[index,:,:,:]
        data = np.zeros((96,WIDTH_BIN))
        for el in range(6):
            data[16*el: (el+1)*16,:] = base[el,:,:]
        snr=(random()*snr_range+snr_base)
      
        cadence, m1,b1 =  new_cadence(data, snr)

        total[0,:,:] = cadence[0:16,:]
        total[1,:,:] = cadence[16:32,:]
        total[2,:,:] = cadence[32:48,:]
        total[3,:,:] = cadence[48:64,:]
        total[4,:,:] = cadence[64:80,:]
        total[5,:,:] = cadence[80:96,:]
    else:
        index = int(plate.shape[0]*random())
        total = plate[index,:,:,:]
    return total


def create_full_cadence(function, samples, snr_base=300, snr_range=10, factor=1, index=1):
    plate = np.load('../../../../../../../datax/scratch/pma/real_filtered/'+str(index)+'.npy')
    data = np.zeros((samples,6,16,WIDTH_BIN))
    for i in prange(data.shape[0]):
        data[i,:,:,:] = function(plate, snr_base=snr_base, snr_range=snr_range, factor=factor) 
    return data

@njit(nopython=True)
def stacking(data):
    return np.array(data)


def create_full_cadence_multicore(function, samples, snr_base=300, snr_range=10, factor=1):
    CORES = 39
    per_batch = samples//CORES
    with Pool(CORES) as p:
        data = p.map(functools.partial(create_full_cadence,function, per_batch, snr_base, snr_range, factor), 
                                range(CORES))
    data = np.vstack(data)
    print(data.shape)
    return data