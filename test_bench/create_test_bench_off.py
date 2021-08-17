import numpy as np
from blimpy import Waterfall
import pandas as pd 
import sys

sys.path.insert(1, '../ML_Training')
sys.path.insert(2, '../GBT_pipeline')
sys.path.insert(3, '../test_bench')
from forest_primer import forest_primer
from execute_model import model_load
import joblib

WIDTH_BIN = 4096

# target = 'HIP15638'
target = 'HIP83043'

list_index = [[1023.9257812499999, 1080.322265625], 
[1080.322265625, 1136.71875], [1136.71875, 1193.115234375], 
[1193.115234375, 1249.51171875], [1249.51171875, 1305.908203125], 
[1305.908203125, 1362.3046875], [1362.3046875, 1418.701171875],
 [1418.701171875, 1475.09765625], [1475.09765625, 1531.494140625], 
 [1531.494140625, 1587.890625], [1587.890625
, 1644.287109375], [1644.287109375, 1700.68359375], 
[1700.68359375, 1757.080078125], [1757.080078125, 1813.4765625], 
[1813.4765625, 1869.873046875], [1869.873046875, 1926.26953125]]

model_name = "VAE-BLPC1-ENCODER_compressed_512v3-10"
model = model_load(model_name+".h5")



for k in range(16):
    f_start, f_stop =list_index[k][0], list_index[k][1]
    resolution = 2.835503418452676e-06

    df =pd.read_csv('../GBT_pipeline/result/'+target +'_directory.csv')
    name = df['0'].tolist()

    # loaded_data = np.zeros((6,16,1,20185088))
    loaded_data = np.zeros((6,16,1,19889408))

    # loaded_data = np.zeros((6,16,1,59668224))

    for i in range(6):
        loaded_data[i,:,:,:] =  Waterfall(name[i],f_start=f_start, f_stop=f_stop).data

    print(loaded_data.shape)
    num_samples = int((f_stop-f_start)//resolution//WIDTH_BIN)

    final_set = np.zeros((num_samples,6, 16,WIDTH_BIN))
    for i in range(num_samples):
        final_set[i,:,:,:] = loaded_data[:,:,0,i*(WIDTH_BIN):(i+1)*WIDTH_BIN]

    print(final_set[:].shape)
    print(final_set.shape[0]//3)

    # forest = joblib.load('../test_bench/random_forest_1000.joblib')
    np.save('../../../../../../../datax/scratch/pma/off_band_test/real_filtered_LARGE_test_'+str(k)+'_'+target+'.npy', final_set[:])