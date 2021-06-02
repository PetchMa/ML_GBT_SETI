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
from decorated_search_multicore import classification_data
from execute_model import model_load

# variable to control how long the search should be in terms of number of files
TOTAL_SEARCHES = 10

# Helps append string to the beginining of a list of strings 
def change(cadence, leading):
    for i in range(len(cadence)):
        cadence[i] = leading+str(cadence[i])
        print(cadence[i])
    return cadence

# read the file for list of cadences
df = pd.read_csv('../data_archive/L_band_directory.csv')
headers = list(df.columns.values[2:])

start_begin =  time.time()
# Load the model into memory
model = model_load("../test_bench/VAE-ENCODERv27.h5")
COUNT = 0
for col in headers:
    start = time.time()
    # create a list of cadence directories
    cadence_set = list(df[col].values)
    # Change the root directory to get the file
    cadence_set = change(cadence_set, '../../../../../../../')
    # Run the search on the data set 
    classification_data(str(col), cadence_set, model, "./", iterations=14)
    print("time: "+str(time.time()-start_begin))
    print("time: execution: "+str(time.time()-start))
    print(cadence_set)
    pd.DataFrame(cadence_set).to_csv(str(col)+"_directory.csv")
    COUNT+=1
    if COUNT == TOTAL_SEARCHES:
        print("DONE")
        break

