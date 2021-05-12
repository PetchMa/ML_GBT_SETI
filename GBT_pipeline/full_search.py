import pandas  as pd
import time 
import sys
sys.path.insert(1, '../ML_Training')
from decorated_search import classification_data
from execute_model import model_load

TOTAL_SEARCHES = 1

def change(cadence, leading):
    for i in range(len(cadence)):
        cadence[i] = leading+str(cadence[i])
        print(cadence[i])
    return cadence


df = pd.read_csv('../data_archive/L_band_directory.csv')
headers = list(df.columns.values[2:])

start_begin =  time.time()
model = model_load("../../VAE-ENCODERv5.h5")
COUNT = 0
for col in headers:
    start = time.time()
    cadence_set = list(df[col].values)
    cadence_set = change(cadence_set, '../../../../../../../')
    data = classification_data(str(col), cadence_set, model, "./", iterations=4)
    print("time: "+str(time.time()-start_begin))
    print("time: execution: "+str(time.time()-start))
    COUNT+=1
    if COUNT == TOTAL_SEARCHES:
        print("DONE")
        break

