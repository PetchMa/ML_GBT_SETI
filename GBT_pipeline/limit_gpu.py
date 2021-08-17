import tensorflow as tf
from tensorflow import keras

def limit_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
            tf.config.experimental.set_virtual_device_configuration(
                gpus[1],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
            tf.config.experimental.set_virtual_device_configuration(
                gpus[2],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
            tf.config.experimental.set_virtual_device_configuration(
                gpus[3],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def limit_gpu_custom(memory_list):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_list[0])])
            tf.config.experimental.set_virtual_device_configuration(
                gpus[1],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_list[1])])
            tf.config.experimental.set_virtual_device_configuration(
                gpus[2],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_list[2])])
            tf.config.experimental.set_virtual_device_configuration(
                gpus[3],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_list[3])])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)