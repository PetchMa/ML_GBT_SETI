import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

def model_load(model_file):
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = load_model(model_file)

def model_predict_distribute(model_file,data):
    model =  model_load(model_file)
    return model.predict(data)
