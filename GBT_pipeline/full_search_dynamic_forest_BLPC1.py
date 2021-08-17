# ============================================================
# Author: Peter Xiangyuan Ma
# Date: May 19 2021
# Purpose: interface with the directories and the list of all the
#targets used for clustering 
# ============================================================

import pandas  as pd
import time 
import sys
from decorated_search_forest import classification_data
sys.path.insert(1, '../ML_Training')

from execute_model import model_load, model_load_custom
import os, psutil
import gc
from intro import intro
from sklearn.tree import DecisionTreeClassifier
import joblib
from limit_gpu import limit_gpu
from forest_primer import forest_primer
import numpy as np

process = psutil.Process(os.getpid())
# variable to control how long the search should be in terms of number of files
TOTAL_SEARCHES = -357+ 715

# Helps append string to the beginining of a list of strings 
def change(cadence, leading):
    for i in range(len(cadence)):
        cadence[i] = leading+str(cadence[i])
        print(cadence[i])
    return cadence
intro(TOTAL_SEARCHES)
# read the file for list of cadences
df = pd.read_csv('../data_archive/fine_cadences_formatted_updated.csv')
# df = pd.read_csv('../data_archive/L_band_directory.csv')

headers = list(df.columns.values[1:])

start_begin =  time.time()
# Load the model into memory
limit_gpu()
# model = model_load("../test_bench/VAE-BLPC1-ENCODER_compressed_512v3-10.h5")
model = model_load_custom("../test_bench/VAE-BLPC1-ENCODER_compressed_512v6-6.h5")

# forest = joblib.load('../test_bench/random_forest_1000.joblib')

forest_set =[]
for i in range(16):
    forest_set.append(joblib.load('../test_bench/random_forest_1000_double.joblib'))


gc.collect()
COUNT = 0
print("Tota`l Number of Files: "+str(len(headers)))
for i in range(357, 715):
    print("index iteration: "+str(i))
    col = headers[i]
    start = time.time()
    # create a list of cadence directories
    cadence_set = list(df[col].values)
    print(cadence_set)
    print(len(cadence_set))
    # Change the root directory to get the file
    cadence_set = change(cadence_set, '../../../../../../..')
    for i in range(len(cadence_set)):
        cadence_set[i] = cadence_set[i].replace(' ','')
    # Run the search on the data set 
    result = classification_data(str(col), cadence_set, model,forest_set, "result_BLPC1/", iterations=16)
    print("time: execution: "+str(time.time()-start))
    
    pd.DataFrame(cadence_set).to_csv("result_BLPC1/"+str(col)+"_directory.csv")
    print("Memory used")
    print(process.memory_info().rss*1e-9)  # in bytes 
    gc.collect()
    COUNT+=1
    # if COUNT == TOTAL_SEARCHES:
    #     print("DONE")
    #     break

