import sys
sys.path.insert(1, '../GBT_pipeline')
from synthetic import create_true, create_full_cadence, create_false, create_true_single_shot
import matplotlib.pyplot as plt
import numpy as np 
from single_search import search
from execute_model import model_load
import tensorflow as tf
tf.get_logger().setLevel('INFO')

NUM_SAMPLES = 10000

print("Loading in plate")

plate = np.load('../../filtered.npy')

print("Creating False")
false_data = create_full_cadence(create_false, plate = plate, samples = NUM_SAMPLES, snr_base=300, snr_range=20)


print("Creating True")
true_data = create_full_cadence(create_true, plate = plate, samples = NUM_SAMPLES,  snr_base=300, snr_range=20, factor =0.1)

# print("Creating Single Shot True")
# true_single_shot = create_full_cadence(create_true_single_shot, plate = plate, samples = 10000, snr_base=300, snr_range=20, factor=10)

print("Load Model")
model = model_load("VAE-ENCODERv9.h5")


print("Search False")
search(false_data, model, False)

print("Search True")
search(true_data, model, True)

# print("Search True Single Shot")
# search(true_single_shot, model, True)
