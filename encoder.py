import tensorflow as tf
from tensorflow import keras 

model = keras.models.load_model('00_6_autoencoder_[fit]_balanced_class-6-bands-wo-loss-morph-control.h5', compile=False)

encoder = keras.Model(model.input, model.get_layer('encoder').output)
print(encoder.summary())
