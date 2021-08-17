import sys
sys.path.insert(1, '../GBT_pipeline')
sys.path.insert(2, '../ML_Training')
from synthetic_real_dynamic import create_true, create_full_cadence, create_false, create_true_single_shot, create_true_faster
import matplotlib.pyplot as plt
import numpy as np 
from single_search import search
from execute_model import model_load
import tensorflow as tf
import pandas as pd
from preprocess_dynamic import resize_par
tf.get_logger().setLevel('INFO')

NUM_SAMPLES = 1000
WIDTH_BIN = 4096
print("Loading in plate")

plate = np.load('../../../../../../../datax/scratch/pma/real_filtered_LARGE_test_HIP15638.npy')
print("Load Model")
model_name = "VAE-BLPC1-ENCODER_compressed_512v4-5"
model = model_load(model_name+".h5")

FACTOR = 1
snr_range=5
snr_list = [20,25,30,35,40,45,50, 55,60, 65,70,75]
results = [] 
for snr in snr_list:
    print(snr)
    print("Creating False")
    false_data = abs((create_false, plate = plate, samples = NUM_SAMPLES, snr_base=snr, snr_range=snr_range, WIDTH_BIN=WIDTH_BIN))

    print("Creating True")
    true_data = create_full_cadence(create_true, plate = plate, samples = NUM_SAMPLES,  snr_base=snr, snr_range=snr_range, factor =FACTOR, WIDTH_BIN=WIDTH_BIN)

    print("Creating Single Shot True")
    true_single_shot = create_full_cadence(create_true_single_shot, plate = plate, samples = NUM_SAMPLES, snr_base=snr, snr_range=5, WIDTH_BIN=WIDTH_BIN)

    #  27 is the good model
    print("Search False")
    false_res = search(resize_par(false_data, factor=8), model, False)

    print("Search True")
    true_res = search(resize_par(true_data, factor=8), model, True)

    print("Search True Single Shot")
    single_true = search(resize_par(true_single_shot, factor=8), model, True)
    results.append([snr,false_res,true_res,single_true])

df = pd.DataFrame(results, columns =['SNR', 'False', 'True', 'Single_True'])   
df.to_csv(model_name+'_full_test.csv')

