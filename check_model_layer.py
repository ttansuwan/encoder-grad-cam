import tensorflow as tf
from tensorflow import keras 
import tensorflow_addons as tfa

def triplet_loss(margin = 1.0):
    def inner_triplet_loss_objective(y_true, y_pred):
        labels = y_true
        embeddings = y_pred
        return tfa.losses.triplet_semihard_loss(y_true = labels, y_pred = embeddings, margin = margin)
    return inner_triplet_loss_objective

model = keras.models.load_model('00_6_autoencoder_[fit]_balanced_class-6-bands-wo-loss-morph-control.h5', compile=False)
print(model.summary())

keras.utils.plot_model(model, to_file='./model.png', show_shapes=True, expand_nested=True)

print('model output', model.output)

encoder = model.get_layer('encoder')
print(encoder.summary())
test = encoder.get_layer('conv2d_6').output
print('test', test)

decoder = model.get_layer('decoder')
print(decoder.summary())

classifier = model.get_layer('classifier')
print(classifier.summary())