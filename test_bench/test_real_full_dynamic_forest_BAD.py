import sys
sys.path.insert(1, '../GBT_pipeline')
sys.path.insert(2, '../ML_Training')
from synthetic_real_dynamic import create_true, create_full_cadence, create_false, create_true_single_shot, create_true_faster
import matplotlib.pyplot as plt
import numpy as np 
from single_search_RF import search
from execute_model import model_load
import tensorflow as tf
import pandas as pd
from preprocess_dynamic import resize_par
import gc 
from forest_primer import forest_primer
import joblib
tf.get_logger().setLevel('INFO')

NUM_SAMPLES = 1000
WIDTH_BIN = 4096
freq_interval = 1

print("Loading in plate train")
# plate1 = np.load('../../../../../../../datax/scratch/pma/real_filtered_LARGE_HIP110750.npy')[12000:16000]
# plate2 = np.load('../../../../../../../datax/scratch/pma/real_filtered_LARGE_HIP13402.npy')[:4000]
# plate3 = np.load('../../../../../../../datax/scratch/pma/real_filtered_LARGE_HIP8497.npy')[:4000]
# plate_train = np.vstack([plate1, plate2, plate3])
# del plate1, plate2, plate3 

# plate_train = np.load('../../../../../../../datax/scratch/pma/real_filtered_LARGE_HIP110750.npy')

print("Load Model")
model_name = "VAE-BLPC1-ENCODER_compressed_512v3-10"
model = model_load(model_name+".h5")
forest = joblib.load('random_forest_1000.joblib')
# forest = forest_primer(model, plate_train)
# del plate_train
gc.collect()
print("Load plate")
# plate = np.load('../../../../../../../datax/scratch/pma/real_filtered_LARGE_test_HIP15638.npy')
plate = np.load('../../../../../../../datax/scratch/pma/off_band_test/real_filtered_LARGE_test_'+str(freq_interval)+'_HIP15638.npy')
# plate = np.load('../../../../../../../datax/scratch/pma/real_filtered_LARGE_test_offband_HIP83043.npy')

thresh=0.5

FACTOR = 1
snr_range=5
snr_list = [20,25,30,35,40,45,50, 55,60, 65,70,75]
results = [] 
for snr in snr_list:
    print(snr)
    print("Creating False")
    false_data = abs(create_full_cadence(create_false, plate = plate, samples = NUM_SAMPLES, snr_base=snr, snr_range=snr_range, WIDTH_BIN=WIDTH_BIN))

    print("Creating True")
    true_data = create_full_cadence(create_true, plate = plate, samples = NUM_SAMPLES,  snr_base=snr, snr_range=snr_range, factor =FACTOR, WIDTH_BIN=WIDTH_BIN)

    print("Creating Single Shot True")
    true_single_shot= create_full_cadence(create_true_single_shot, plate = plate, samples = NUM_SAMPLES, snr_base=snr, snr_range=5, WIDTH_BIN=WIDTH_BIN)

    #  27 is the good model
    print("Search False")
    false_res, mean_false = search(resize_par(false_data, factor=8), model,forest, False, thresh)

    print("Search True")
    true_res, mean_true = search(resize_par(true_data, factor=8), model,forest, True, thresh)

    print("Search True Single Shot")
    single_true, mean_true_shot  = search(resize_par(true_single_shot, factor=8), model,forest, True, thresh)
    results.append([snr,false_res,true_res,single_true, mean_false, mean_true, mean_true_shot])

df = pd.DataFrame(results, columns =['SNR', 'False', 'True', 'Single_True', 'ave_false_prob', 'avg_true_prob', 'avg_true_shot'])   
df.to_csv(model_name+'diff_full_test_forest_'+str(thresh)+'.csv')

