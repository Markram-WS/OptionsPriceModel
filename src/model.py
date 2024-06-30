from tensorflow import keras
import tensorflow as tf
from   src.utils import debug
import numpy as np
import wandb
#https://keras.io/examples/generative/molecule_generation/

class OptionChainGenerator(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.optVal_loss_tracker = keras.metrics.Mean(name="optVal_loss")
        self.vol_loss_tracker = keras.metrics.Mean(name="vol_loss")
        
    def train_step(self, data):
        X_input, Y_real = data  

        with tf.GradientTape() as tape:
            #__call__ objects
            z_mean, z_log_var, c_bid, c_ask, c_volume, p_bid, p_ask, p_volume = self(
                X_input, training=True
            )
            generated_data = [c_bid, c_ask, c_volume, p_bid, p_ask, p_volume]
            total_loss,kl_loss  ,optVal_loss, volume_loss= self._compute_loss(
                z_mean, z_log_var, Y_real, generated_data
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.optVal_loss_tracker.update_state(optVal_loss)
        self.vol_loss_tracker.update_state(volume_loss)

        return {"total_loss": self.total_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                "optVal_loss": self.optVal_loss_tracker.result(),
                "vol_loss": self.vol_loss_tracker.result()
                }

        
    def _compute_loss(self,z_mean, log_var, Y_real, generated_data):
        features_real = Y_real
        # Reconstruction loss for each output
        reconstruction_losses = []
        reconstruction_volume_losss = []
        reconstruction_values_losss = []
        colList = ["c_bid", "c_ask", "c_volume", "p_bid", "p_ask", "p_volume"]
        for input_name, decoder_output in zip(colList, generated_data):
            
            if input_name == "p_volume":
                # Compute reconstruction loss for p_volume
                #! features_input index of p_volume
                reconstruction_loss_p_volume = tf.reduce_mean(tf.square(tf.reduce_sum(decoder_output, axis=1, keepdims=True)  - tf.reduce_sum(features_real[:, :, -1], axis=1, keepdims=True)))
                reconstruction_volume_losss.append(reconstruction_loss_p_volume)
                reconstruction_losses.append(reconstruction_loss_p_volume)
                
            elif input_name == "c_volume":
                # Compute reconstruction loss for c_volume
                #! features_input index of c_volume
                reconstruction_loss_c_volume = tf.reduce_mean(tf.square(tf.reduce_sum(decoder_output, axis=1, keepdims=True)  - tf.reduce_sum(features_real[:, :, -4], axis=1, keepdims=True)))
                reconstruction_volume_losss.append(reconstruction_loss_c_volume)
                reconstruction_losses.append(reconstruction_loss_c_volume)
                
            else:
                # Compute reconstruction loss for other features
                reconstruction_loss = tf.reduce_mean(tf.square(tf.reduce_sum(decoder_output, axis=1, keepdims=True)  - tf.reduce_sum(features_real[:, :, colList.index(input_name)], axis=1, keepdims=True)))
                reconstruction_values_losss.append(reconstruction_loss)
                reconstruction_losses.append(reconstruction_loss)

        # Reconstruction loss: sum of all reconstruction losses
        reconstruction_loss_total = tf.reduce_sum(reconstruction_losses)
        reconstruction_volume_loss_total = tf.reduce_sum(reconstruction_volume_losss)
        reconstruction_values_losss = tf.reduce_sum(reconstruction_values_losss)
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(z_mean) - tf.exp(log_var), axis=-1)

        # Total loss
        total_loss = tf.reduce_mean(reconstruction_loss_total + kl_loss)

        return total_loss,kl_loss,reconstruction_values_losss,reconstruction_volume_loss_total

    def call(self, inputs, training=False):
        z_mean, log_var = self.encoder(inputs)
        z = Sampling()([z_mean, log_var])

        c_bid,c_ask,c_volume,p_bid,p_ask,p_volume = self.decoder(z)

        return z_mean, log_var, c_bid, c_ask, c_volume, p_bid, p_ask, p_volume

    
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_log_var)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    
def create_adjacency_matrix(options, threshold=5):
    """
    สร้าง adjacency matrix สำหรับ Options Chain โดยใช้ความใกล้เคียงของ strike price และ DTE
    """
    num_options = len(options)
    adjacency_matrix = np.zeros((num_options, num_options))

    for i in range(num_options):
        for j in range(i, num_options):
            # พิจารณาเชื่อมโยงระหว่าง options หาก strike price ต่างกันไม่เกิน threshold และ DTE เท่ากัน
            if abs(options[i]['strike_price'] - options[j]['strike_price']) <= threshold and options[i]['dte'] == options[j]['dte']:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    return adjacency_matrix