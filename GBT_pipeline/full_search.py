# ============================================================
# Author: Peter Xiangyuan Ma
# Date: May 19 2021
# Purpose: interface with the directories and the list of all the
#targets used for clustering 
# ============================================================

import pandas  as pd
import time 
import sys
sys.path.insert(1, '../ML_Training')
from decorated_search_multicore_batch import classification_data
from execute_model import model_load
import os, psutil
import gc
from intro import intro
process = psutil.Process(os.getpid())
# variable to control how long the search should be in terms of number of files
TOTAL_SEARCHES = 10

# Helps append string to the beginining of a list of strings 
def change(cadence, leading):
    for i in range(len(cadence)):
        cadence[i] = leading+str(cadence[i])
        print(cadence[i])
    return cadence
intro(TOTAL_SEARCHES)
# read the file for list of cadences
df = pd.read_csv('../data_archive/L_band_directory.csv')
headers = list(df.columns.values[2:])

start_begin =  time.time()
# Load the model into memory
model = model_load("../test_bench/VAE-ENCODERv27.h5")
COUNT = 0
print("Total Number of Files: "+str(len(headers)))
for i in range(30, len(headers)):
    col = headers[i]
    start = time.time()
    # create a list of cadence directories
    cadence_set = list(df[col].values)
    print(cadence_set)
    # Change the root directory to get the file
    cadence_set = change(cadence_set, '../../../../../../../')
    # Run the search on the data set 
    result = classification_data(str(col), cadence_set, model, "result/", iterations=16)
    print("time: execution: "+str(time.time()-start))
    
    pd.DataFrame(cadence_set).to_csv("result/"+str(col)+"_directory.csv")
    print("Memory used")
    print(process.memory_info().rss*1e-9)  # in bytes 
    gc.collect()
    COUNT+=1
    if COUNT == TOTAL_SEARCHES:
        print("DONE")
        break

