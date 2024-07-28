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
        
    def compile(self, optimizer):
        super(OptionChainGenerator, self).compile()
        self.optimizer = optimizer
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        # self.val_loss_tracker = tf.keras.metrics.Mean(name="values_loss")
        # self.vol_loss_tracker = tf.keras.metrics.Mean(name="volume_loss")
        
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.val_kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        # self.val_values_loss_tracker = tf.keras.metrics.Mean(name="values_loss")
        # self.val_volume_loss_tracker = tf.keras.metrics.Mean(name="volume_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.kl_loss_tracker,
            # self.val_loss_tracker,
            # self.vol_loss_tracker,
            self.val_total_loss_tracker,
            self.val_kl_loss_tracker,
            # self.val_values_loss_tracker,
            # self.val_volume_loss_tracker
        ]
        
    def train_step(self, data):
        X_input, Y_input = data  

        with tf.GradientTape() as tape:
            #__call__ objects
            z_mean, z_log_var, gen_features  = self(
                X_input, training=True
            )
            kl_loss,total_loss = self._compute_loss(
                z_mean, z_log_var, Y_input, gen_features
            )            

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(grad, -1, 1) for grad in grads if grad is not None]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        # self.val_loss_tracker.update_state(values_loss)
        # self.vol_loss_tracker.update_state(volume_loss)

        return {"kl_loss": self.kl_loss_tracker.result(),
                "total_loss": self.total_loss_tracker.result(),
                # "values_loss":self.val_loss_tracker.result(),
                # "volume_loss":self.vol_loss_tracker.result()
                }
        
        

    def test_step(self, data):
        X_input, Y_real = data  

        z_mean, z_log_var, gen_features = self(X_input, training=False)
        kl_loss,total_loss = self._compute_loss(z_mean, z_log_var, Y_real, gen_features,mode='val')

        self.val_kl_loss_tracker.update_state(kl_loss)
        self.val_total_loss_tracker.update_state(total_loss)
        # self.val_values_loss_tracker.update_state(values_loss)
        # self.val_volume_loss_tracker.update_state(volume_loss)

        return {"kl_loss": self.val_kl_loss_tracker.result(),
                "total_loss": self.val_total_loss_tracker.result(),
                # "values_loss":self.val_values_loss_tracker.result(),
                # "volume_loss":self.val_volume_loss_tracker.result()
                }
        
    def _compute_loss(self,z_mean, log_var, y_real, generated_data,mode='Train'):
        # Mean Squared Error loss function
        mse = tf.keras.losses.MeanSquaredError()
        # Calculate the loss for the features
        reconstruction_loss = mse(y_real, generated_data)
        # KL divergence loss
        log_var_clipped = tf.clip_by_value(log_var, -10, 10)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var_clipped - tf.square(z_mean) - tf.exp(log_var_clipped), axis=-1))

        # # Total loss
        total_loss = reconstruction_loss  + kl_loss
        return kl_loss,total_loss

    # def vae_loss(self,y_true, y_pred):
    #     kl_loss, total_loss = self._compute_loss(y_true, y_pred, self.encoder.get_layer('z_mean').output, self.encoder.get_layer('z_log_var').output)
    #     return total_loss

    def call(self, inputs, training=False):
        z_mean, log_var = self.encoder(inputs)
        z = Sampling()([z_mean, log_var])

        #c_bid,c_ask,c_volume,p_bid,p_ask,p_volume = self.decoder(z)
        gen_features = self.decoder(z)
        return z_mean, log_var, gen_features

    
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