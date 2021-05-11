import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import VAE
from Sampling import Sampling


# Build model in two forms one in distributed GPU training 
# Model is a VAE model with some disentanglement 
# recreate the model with the following model 
def build_model(latent_dim = 6,dens_lay = 1024, kernel = (3,3),conv1 = 0,conv2 = 0,conv3 = 0,conv4 = 0, lr= 0.0005 ):
    encoder_inputs = keras.Input(shape=(16, 256, 1))
    x = layers.BatchNormalization()(encoder_inputs)
    x = layers.Conv2D(16, kernel, activation="relu", strides=2, padding="same")(x)
    for i in range(conv1):
        x = layers.Conv2D(16, kernel, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
    x = layers.Conv2D(32, kernel, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    for i in range(conv2):
        x = layers.Conv2D(32, kernel, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, kernel, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    for i in range(conv3):
        x = layers.Conv2D(64, kernel, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, kernel, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    for i in range(conv4):
        x = layers.Conv2D(128, kernel, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dens_lay, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(dens_lay, activation="relu")(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1 * 16 * 128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((1, 16, 128))(x)
    
    x = layers.Conv2DTranspose(128, kernel, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    for i in range(conv4):
        x = layers.Conv2DTranspose(128, kernel, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
    
    for i in range(conv3):
        x = layers.Conv2DTranspose(64, kernel, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, kernel, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    for i in range(conv2):
        x = layers.Conv2DTranspose(32, kernel, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, kernel, activation="relu", strides=2, padding="same")(x)
    
    
    for i in range(conv1):
        x = layers.Conv2DTranspose(16, kernel, activation="relu", strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(16, kernel, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    decoder_outputs = layers.Conv2DTranspose(1, kernel, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(lr))
    return vae

#distribut GPU train with a mirrored strategy
def multi_gpu_load():
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = build_model()
    return model