import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from Sampling import Sampling
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def sample_creation(inputs):
    z_mean = inputs[0]
    z_log_var = inputs[1]
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def model_load(model_file):
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice())
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = load_model(model_file, custom_objects={'Sampling': Sampling})
    return model

def model_load_custom(model_file):
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1", "/gpu:2","/gpu:3"],
        cross_device_ops=tf.distribute.ReductionToOneDevice())
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = load_model(model_file, custom_objects={'Sampling': Sampling})
    return model

def model_predict_distribute(model_file,data):
    model =  model_load(model_file)
    return model.predict(data)

def model_predict_distribute_vae(model_file,data):
    model =  model_load(model_file)
    return sample_creation(model.predict(data))