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
            generated_data = [gen_features]
            kl_loss,total_loss = self._compute_loss(
                z_mean, z_log_var, Y_input, generated_data
            )            

        grads = tape.gradient(total_loss, self.trainable_weights)
        #grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads if grad is not None]
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
        generated_data = [gen_features]
        kl_loss,total_loss = self._compute_loss(z_mean, z_log_var, Y_real, generated_data,mode='val')

        self.val_kl_loss_tracker.update_state(kl_loss)
        self.val_total_loss_tracker.update_state(total_loss)
        # self.val_values_loss_tracker.update_state(values_loss)
        # self.val_volume_loss_tracker.update_state(volume_loss)

        return {"kl_loss": self.val_kl_loss_tracker.result(),
                "total_loss": self.val_total_loss_tracker.result(),
                # "values_loss":self.val_values_loss_tracker.result(),
                # "volume_loss":self.val_volume_loss_tracker.result()
                }
        
    def _compute_loss(self,z_mean, log_var, Y_real, generated_data,mode='Train'):
        # Reconstruction loss for each output
        reconstruction_losses = []
        reconstruction_volume_losss = []
        reconstruction_values_losss = []
        #colList = ["c_bid", "c_ask", "c_volume", "p_bid", "p_ask", "p_volume"]
        colList = ["features"]
        
        
        genDataVol = []
        realVol = []
        mse = tf.keras.losses.MeanSquaredError()
        for  col,genData in zip(colList,generated_data):
            #reconstruction_loss  = tf.reduce_mean(tf.square(tf.reduce_sum(decoder_output, axis=1, keepdims=True)  - tf.reduce_sum(features_real[:, :, colList.index(input_name)], axis=1, keepdims=True)))
       
            # Calculate the loss for the other features normally
            #reconstruction_loss = tf.reduce_mean( tf.square(genData - tf.expand_dims(Y_real[:, :, colList.index(col)], axis=-1) ) )
            reconstruction_loss  = mse(tf.expand_dims(Y_real[:, :, colList.index(col)], axis=-1),genData)

            
            if col == "p_volume":
                # Compute reconstruction loss for p_volume
                #! features_input index of p_volume
                #genDataVol.append( genData) 
                #realVol.append(tf.expand_dims(Y_real[:, :, colList.index(col)], axis=-1)  ) 
                reconstruction_volume_losss.append(reconstruction_loss)
            elif col == "c_volume":
                # Compute reconstruction loss for c_volume
                #! features_input index of c_volume
                reconstruction_volume_losss.append(reconstruction_loss)
                #genDataVol.append(genData)
                #realVol.append(tf.expand_dims(Y_real[:, :, colList.index(col)], axis=-1))
            else:
                reconstruction_values_losss.append(reconstruction_loss)
                
            tf.debugging.check_numerics(genData, message=f'{mode} {col} genData contains NaNs or Infs - {genData}')
            tf.debugging.check_numerics(tf.expand_dims(Y_real[:, :, colList.index(col)], axis=-1), message=f'{mode} {col} [Y] NaNs or Infs - {genData}')
        # Reconstruction loss: sum of all reconstruction losses
        #reconstruction_volume_losss = tf.reduce_mean( tf.square(tf.reduce_sum(genDataVol) - tf.reduce_sum(realVol) ) )
        #reconstruction_volume_total = tf.reduce_sum(reconstruction_volume_losss)  #minimize the importance
        print(reconstruction_values_losss)
        reconstruction_values_total= tf.reduce_sum(reconstruction_values_losss)  #minimize the importance
        print(reconstruction_values_total)
        # KL divergence loss
        log_var = tf.clip_by_value(log_var, -1.0, 1.0)
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(z_mean) - tf.exp(log_var), axis=-1)
        print(reconstruction_values_total  + kl_loss)
        # # Total loss
        total_loss = reconstruction_values_total  + kl_loss
        # volume_loss = tf.reduce_mean(reconstruction_volume_total*0.4 + kl_loss)
        # values_loss = tf.reduce_mean(reconstruction_values_total + kl_loss)
        

  
        return kl_loss,total_loss

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