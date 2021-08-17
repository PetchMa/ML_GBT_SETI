import sys
sys.path.insert(1, '../GBT_pipeline')
from synthetic_real import create_true, create_full_cadence, create_false, create_true_single_shot, create_true_faster
import matplotlib.pyplot as plt
import numpy as np 
from single_search import search
from execute_model import model_load
import tensorflow as tf
tf.get_logger().setLevel('INFO')

NUM_SAMPLES = 1000

print("Loading in plate")

plate = np.load('../../real_filtered_test.npy')[:77692-5]


print("Creating False")
false_data = abs(create_full_cadence(create_false, plate = plate, samples = NUM_SAMPLES, snr_base=20, snr_range=10))

print("Creating True")
true_data = create_full_cadence(create_true, plate = plate, samples = NUM_SAMPLES,  snr_base=20, snr_range=10, factor =10)

print("Creating Single Shot True")
true_single_shot = create_full_cadence(create_true_single_shot, plate = plate, samples = NUM_SAMPLES, snr_base=20, snr_range=10)



print("Load Model")
model = model_load("VAE-ENCODERvmini_52.h5")
#  27 is the good model
print("Search False")
search(false_data, model, False)

print("Search True")
search(true_data, model, True)

print("Search True Single Shot")
search(true_single_shot, model, True)
