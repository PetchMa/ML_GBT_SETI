import sys
sys.path.insert(1, '../GBT_pipeline')
from synthetic_real import create_true, create_full_cadence, create_false, create_true_single_shot, create_true_faster
import matplotlib.pyplot as plt
import numpy as np 
from single_search import search
from execute_model import model_load
import tensorflow as tf
import pandas as pd
tf.get_logger().setLevel('INFO')

NUM_SAMPLES = 1000

print("Loading in plate")

plate = np.load('../../real_filtered_test.npy')[:77692-5]
print("Load Model")
model = model_load("VAE-ENCODERvmini_59_true.h5")

FACTOR = 10
snr_list = [10,15,20,25,30,35,40]
results = []
for snr in snr_list:
    print("Creating False")
    false_data = abs(create_full_cadence(create_false, plate = plate, samples = NUM_SAMPLES, snr_base=snr, snr_range=5))

    print("Creating True")
    true_data = create_full_cadence(create_true, plate = plate, samples = NUM_SAMPLES,  snr_base=snr, snr_range=5, factor =FACTOR)

    print("Creating Single Shot True")
    true_single_shot = create_full_cadence(create_true_single_shot, plate = plate, samples = NUM_SAMPLES, snr_base=snr, snr_range=5)

    #  27 is the good model
    print("Search False")
    false_res = search(false_data, model, False)

    print("Search True")
    true_res = search(true_data, model, True)

    print("Search True Single Shot")
    single_true = search(true_single_shot, model, True)
    results.append([snr,false_res,true_res,single_true])

df = pd.DataFrame(results, columns =['SNR', 'False', 'True', 'Single_True'])   
df.to_csv('full_test.csv')

