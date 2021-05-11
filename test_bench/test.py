from synthetic import create_true, create_full_cadence, create_false, create_true_single_shot
import matplotlib.pyplot as plt
import numpy as np 


plate = np.load('../../filtered.npy')

true_data = create_full_cadence(create_false, plate = plate, samples = 2, snr_base=300, snr_range=10)
true_data = create_full_cadence(create_true, plate = plate, samples = 2, snr_base=300, snr_range=10, factor =10)

true_data = create_full_cadence(create_true_single_shot, plate = plate, samples = 2, snr_base=300, snr_range=10)


