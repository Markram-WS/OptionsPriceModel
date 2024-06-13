from tensorflow import keras
import tensorflow as tf
#https://keras.io/examples/generative/molecule_generation/
class OptionChainGenerator(keras.Model):
    def __init__(self, encoder, decoder, max_len, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.property_prediction_layer = keras.layers.Dense(1)
        self.max_len = max_len

        self.train_total_loss_tracker = keras.metrics.Mean(name="train_total_loss")
        self.optVal_total_loss_tracker = keras.metrics.Mean(name="optVal_total_loss")
        self.c_vol_total_loss_tracker = keras.metrics.Mean(name="c_vol_total_loss")
        self.p_vol_total_loss_tracker = keras.metrics.Mean(name="p_vol_total_loss")
    def train_step(self, data):
        adjacency_tensor, feature_tensor, qe_tensor = data[0]
        graph_real = [adjacency_tensor, feature_tensor]
        self.batch_size = tf.shape(qe_tensor)[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, qed_pred, gen_adjacency, gen_features = self(
                graph_real, training=True
            )
            graph_generated = [gen_adjacency, gen_features]
            total_loss ,optVal,p_volume_loss, c_volume_loss= self._compute_loss(
                z_log_var, z_mean, qe_tensor, qed_pred, graph_real, graph_generated
            )
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.train_total_loss_tracker.update_state(total_loss)
        self.optVal_total_loss_tracker.update_state(optVal)
        self.c_vol_total_loss_tracker.update_state(c_volume_loss)
        self.p_vol_total_loss_tracker.update_state(p_volume_loss)
        return {"loss": self.train_total_loss_tracker.result(),
                "optVal": self.optVal_total_loss_tracker.result(),
                "c_vol": self.c_vol_total_loss_tracker.result(),
                "p_vol": self.p_vol_total_loss_tracker.result()
                }

        
    def _compute_loss(encoder, decoder, inputs):
        adjacency_input, features_input = inputs

        # Forward pass through the encoder to get latent variables
        z_mean, log_var = encoder([adjacency_input, features_input])
        z = Sampling([z_mean, log_var])

        # Forward pass through the decoder to generate outputs
        decoder_outputs = decoder(z)

        # Reconstruction loss for each output
        reconstruction_losses = []
        reconstruction_p_volume_losss = []
        reconstruction_c_volume_losss = []
        reconstruction_values_losss = []
        for input_name, decoder_output in zip(["c_bid", "c_ask", "c_volume", "p_bid", "p_ask", "p_volume"], decoder_outputs[0]):
            if input_name == "p_volume":
                # Compute reconstruction loss for p_volume
                #! features_input index of p_volume
                reconstruction_loss_p_volume = tf.reduce_mean(tf.square(tf.reduce_sum(decoder_output, axis=1, keepdims=True)  - tf.reduce_sum(features_input[:, :, 0], axis=1, keepdims=True)))
                reconstruction_losses.append(reconstruction_loss_p_volume)
                reconstruction_p_volume_losss.append(reconstruction_loss_p_volume)
            elif input_name == "c_volume":
                # Compute reconstruction loss for c_volume
                #! features_input index of c_volume
                reconstruction_loss_c_volume = tf.reduce_mean(tf.square(tf.reduce_sum(decoder_output, axis=1, keepdims=True)  - tf.reduce_sum(features_input[:, :, 0], axis=1, keepdims=True)))
                reconstruction_losses.append(reconstruction_loss_c_volume)
                reconstruction_c_volume_losss.append(reconstruction_loss_c_volume)
            else:
                # Compute reconstruction loss for other features
                reconstruction_loss = tf.reduce_mean(tf.square(decoder_output - features_input[:, :, 0]))
                reconstruction_losses.append(reconstruction_loss)
                reconstruction_values_losss(reconstruction_loss)

        # Reconstruction loss: sum of all reconstruction losses
        reconstruction_loss_total = tf.reduce_sum(reconstruction_losses)
        reconstruction_p_volume_loss_total = tf.reduce_sum(reconstruction_p_volume_losss)
        reconstruction_c_volume_loss_total = tf.reduce_sum(reconstruction_c_volume_losss)
        reconstruction_values_losss = tf.reduce_sum(reconstruction_values_losss)
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(z_mean) - tf.exp(log_var), axis=-1)

        # Total loss
        total_loss = tf.reduce_mean(reconstruction_loss_total + kl_loss)

        return total_loss,reconstruction_values_losss,reconstruction_p_volume_loss_total,reconstruction_c_volume_loss_total

    def _gradient_penalty(self, graph_real, graph_generated):
        # Unpack graphs
        adjacency_real, features_real = graph_real
        adjacency_generated, features_generated = graph_generated

        # Generate interpolated graphs (adjacency_interp and features_interp)
        alpha = tf.random.uniform([self.batch_size])
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
        adjacency_interp = (adjacency_real * alpha) + (1 - alpha) * adjacency_generated
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
        features_interp = (features_real * alpha) + (1 - alpha) * features_generated

        # Compute the logits of interpolated graphs
        with tf.GradientTape() as tape:
            tape.watch(adjacency_interp)
            tape.watch(features_interp)
            _, _, logits, _, _ = self(
                [adjacency_interp, features_interp], training=True
            )

        # Compute the gradients with respect to the interpolated graphs
        grads = tape.gradient(logits, [adjacency_interp, features_interp])
        # Compute the gradient penalty
        grads_adjacency_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
        grads_features_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
        return tf.reduce_mean(
            tf.reduce_mean(grads_adjacency_penalty, axis=(-2, -1))
            + tf.reduce_mean(grads_features_penalty, axis=(-1))
        )


    def call(self, inputs):
        z_mean, log_var = self.encoder(inputs)
        z = Sampling()([z_mean, log_var])

        gen_adjacency, gen_features = self.decoder(z)

        property_pred = self.property_prediction_layer(z_mean)

        return z_mean, log_var, property_pred, gen_adjacency, gen_features
    
    
    
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