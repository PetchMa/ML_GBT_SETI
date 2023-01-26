# Top 8 Candidates and Data Files
This directory contains the original data of the top 8 candidates cited in our Nature Astronomy Paper.
Please find the original data saved in Python `pickle` which is a dictionary containing keys labeled by `mlc1`, `mlc2`, ..., `mlc8`. Please find the [file here](https://github.com/PetchMa/ML_GBT_SETI/blob/4096_pipeline/data_archive/top_8_data.pkl). 

We created a short schematic to depict the data structure as its saved 

![schematic](https://raw.githubusercontent.com/PetchMa/ML_GBT_SETI/4096_pipeline/data_archive/top_8_cand_struct.jpeg)

Simply put, this is a nested dictionary, such that its keys are the `mlc<x>` for each of the 8. The value of each key is a dictionary so for each candidate it contains `data` , `frequency`  and `all_headers` as the keys. Below is python code to open and plot the data. 

```python 
import pickle
import numpy as np
from blimpy import Waterfall
import matplotlib.pyplot as plt

with open('top_8_data.pkl', 'rb') as handle:
    data = pickle.load(handle)
    

def plot_cadence(data):
    fig, ax = plt.subplots(6, 1, figsize=(10,10))
    fig.subplots_adjust(hspace = .05, wspace=.001)
    for i in range(6):
    # note we need to reverse frequency axis to create the plots in the paper!
        ax[i].imshow(data[i, :, 0, ::-1], aspect=30, vmin = data.min(), vmax = data.max())
        if i==5:
            ax[i].set_xlabel('Frequency bins', fontsize=20)
    fig.text(0.06, 0.5, 'Time bins', ha='center', va='center', rotation='vertical',  fontsize=16)
    plt.show()
    
    
plot_cadence(data['mlc8']['data'])   
```
