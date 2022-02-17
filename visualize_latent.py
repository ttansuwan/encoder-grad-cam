import tensorflow as tf
from tensorflow import keras 

model = keras.models.load_model('AE_out_weights10.h5')
print(model.summary())

encoder = model.get_layer('encoder')
print(encoder.summary())

avg_pooling_layer = encoder.get_layer('average_pooling2d_1').get_weights()
print(avg_pooling_layer)