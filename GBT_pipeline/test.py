import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
sys.path.insert(1, '../ML_Training')
# from decorated_search import classification_data
from decorated_search_multicore import classification_data
from execute_model import model_load
import time

import warnings
warnings.filterwarnings("ignore")


cadence_set = ['../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_58929_GJ380_fine.h5',
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_59291_HIP48887_fine.h5",
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_59650_GJ380_fine.h5",
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_60004_HIP48924_fine.h5",
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_60354_GJ380_fine.h5",
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_60706_HIP48954_fine.h5"
                ]
model = model_load("../../VAE-ENCODERv6.h5")

start=  time.time()
data = classification_data("GJ380", cadence_set, model, "./", iterations=4)
print("time: "+str(time.time()-start))