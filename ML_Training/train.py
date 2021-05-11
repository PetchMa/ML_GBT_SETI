import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from build_model import multi_gpu_load
import random
import datetime


#There is a specific training loop that trains the model to map to same features 
#by looping through a cadence of data. 

def train_loop(model_file, data, batch = 5000, epoch_lower = 1 , epoch = 1000):
    time = str(datetime.datetime.now())
    model = multi_gpu_load(model_file)
    for i in range(epoch):
        index = int(random.random()*6)
        print("Epoch: "+str(i)+ " - fix cadence: "+str(index))
        model.fit(data[:,index,:,:,:],data[:,3,:,:,:], epochs=epoch_lower, batch_size=batch)

    model.encoder.save("../../VAE-ENCODER-"+str(time)+".h5")
    model.decoder.save("../../VAE-DECODER-"+str(time)+".h5")   
    return model