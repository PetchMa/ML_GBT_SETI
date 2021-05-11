import numpy as np
import sys
sys.path.insert(1, '../ML_Training')
from execute_model import model_predict_distribute
from preprocess import get_data

from numba import jit, prange, njit
from blimpy import Waterfall
import time
import random
from sklearn.cluster import SpectralClustering


def 


def classification_data(cadence, model, out_dir, iterations=6):
    header = Waterfall(cadence[0]).header
    end = header['fch1']
    start = header['fch1']+ header['nchans']*header['foff']
    interval = (end-start)/iterations
    WINDOW_SIZE = abs(256*header['foff'])

    freq_ranges = []
    for i in range(iterations):
        f_start = start+i *iterations
        f_stop = start+i *(iterations+1)
        freq_ranges.append([f_start, f_stop])
    
    #execution looop:
    for index in range(iterations):
        data = get_data(cadence,start =freq_ranges[i][0],end =freq_ranges[i][1])
        result = model.predict(data)
