from operator import index
from random import random
import tensorflow as tf
import tensorflow_addons as tfa
from tf_explain.core.grad_cam import GradCAM
import numpy as np
# tf.compat.v1.disable_eager_execution() # enable T1  


class GradCam:
    def __init__(self, model_path, data_path, label_path):
        self.data = self.load_data(data_path, label_path, med_class=1)
        self.model = self.load_model(model_path)

    def find_mice_session(self, y):
        # check label of each mouse
        mice_session = np.unique(y, axis=1).flatten()
        return mice_session

    def random_pick(self, index_session):
        mouse_index = np.random.choice(len(index_session), 1)
        return index_session[mouse_index]

    def convert_np(self, data):
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        return data

    def load_data(self, data_path, label_path, med_class):
        X = np.load(data_path)
        y = np.load(label_path)

        mice_session = self.find_mice_session(y)
        index_session = np.argwhere(mice_session == med_class).flatten()
        mouse_index = self.random_pick(index_session)
        mouse_session = X[mouse_index, 0]

        # select one session 
        mouse_session = np.swapaxes(mouse_session, 2, 3)
        mouse_session = self.convert_np(mouse_session)
        return mouse_session

    def inner_triplet_loss_objective(self, y_true, y_pred):
        labels = y_true
        embeddings = y_pred
        return tfa.losses.triplet_semihard_loss(y_true = labels, y_pred = embeddings, margin = 1.0)

    def predict(self, model, data):
        result = model.predict(data) # add steps=1 when T1 enable
        return result
    
    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path, custom_objects={"inner_triplet_loss_objective": self.inner_triplet_loss_objective})
        
        en_input, en_model, de_model, classifier = model.layers
        en_input = en_input.output
        latent = en_model(en_input)
        z = classifier(latent)
        sliced_model = tf.keras.models.Model(inputs=[en_input], outputs=[z])

        return sliced_model

    def make_model(self, model):
        layers = [l for l in model.layers]
        x = layers[0].output
        for i in range(1, len(layers)):
            x = layers[i](x)

        new_model = tf.keras.models.Model(inputs=layers[0].output, outputs=x)
        return new_model

    def join_model(self, model1, model2):
        layers_model1 = [l for l in model1.layers]

        x = layers_model1[0].output
        for i in range(1, len(layers_model1)):
            x = layers_model1[i](x)
        x = model2(x)
        
        new_model = tf.keras.models.Model(inputs=layers_model1[0].input, outputs=x)
        return new_model

    def make_gradcam_heatmap(self):
        self.model.layers[-1].activation = None
        
        encoder = self.model.get_layer('encoder')
        sliced_model = tf.keras.models.Model(inputs=encoder.inputs, outputs=encoder.outputs[0])
        sliced_model = self.join_model(sliced_model, self.model.get_layer('classifier'))
        print(self.predict(sliced_model, self.data))
        print(sliced_model.summary())

        last_conv_layer_output = sliced_model.get_layer('conv2d_6').output
        grad_model = tf.keras.models.Model(inputs=sliced_model.inputs, outputs=[last_conv_layer_output, sliced_model.output])
        
        with tf.GradientTape() as tape:
            tape.watch(self.data)
            last_conv_layer_output, preds = grad_model(self.data)
            pred_index = tf.math.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, self.data) 
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def gradcam(self): # not inuse
        preds = self.predict(self.model, self.data)
        class_idx = np.argmax(preds[0])
        class_output = preds[:, class_idx]
        print(class_output)
        last_conv_layer = self.model.get_layer('encoder').get_layer('conv2d_6').get_output_at(-1)
        grads = tf.keras.backend.gradients(class_output, last_conv_layer)[0]
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

if __name__ == "__main__":
    MODEL_PATH = "./00_6_autoencoder_[fit]_balanced_class-6-bands-wo-loss-morph-control.h5"
    DATA_PATH = "./X_meth_control.npy"
    LABEL_PATH = "./Y_meth_control.npy"
    grad_cam = GradCam(MODEL_PATH, DATA_PATH, LABEL_PATH)
    grad_cam.make_gradcam_heatmap()