from tensorflow import keras
import tensorflow as tf

class RelationalGraphConvLayer(keras.layers.Layer):
    def __init__(
        self,
        units=128,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]

        self.kernel = self.add_weight(
            shape=(bond_dim, atom_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="W",
            dtype=tf.float32,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(bond_dim, 1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="b",
                dtype=tf.float32,
            )

        self.built = True

    def call(self, inputs, training=False):
        adjacency, features = inputs
        # Aggregate information from neighbors
        x = tf.matmul(adjacency, features[:, None, :, :])
        # Apply linear transformation
        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x += self.bias
        # Reduce bond types dim
        x_reduced = tf.reduce_sum(x, axis=1)
        # Apply non-linear transformation
        return self.activation(x_reduced)


def encoder(
    gconv_units, latent_dim, 
    avgVol_shape, current_price_shape, strike_price_shape, dte_shape, #-- frture --
    dense_units, dropout_rate
):
    avgVol = keras.layers.Input(shape=avgVol_shape, name="avgVolume")
    current_price = keras.layers.Input(shape=current_price_shape, name="currentPrice")
    strike_price = keras.layers.Input(shape=strike_price_shape, name="strikePrice")
    dte = keras.layers.Input(shape=dte_shape, name="DateToExpired")

    # Propagate through one or more graph convolutional layers
    for units in gconv_units:
        features_transformed = RelationalGraphConvLayer(units)(
            [avgVol, current_price, strike_price, dte]
        )
    # Reduce 2-D representation of molecule to 1-D
    x = keras.layers.GlobalAveragePooling1D()(features_transformed)

    # Propagate through one or more densely connected layers
    for units in dense_units:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    z_mean = keras.layers.Dense(latent_dim, dtype="float32", name="z_mean")(x)
    log_var = keras.layers.Dense(latent_dim, dtype="float32", name="log_var")(x)

    encoder = keras.Model([avgVol, current_price, strike_price, dte], [z_mean, log_var], name="encoder")

    return encoder


def decoder(dense_units, dropout_rate, latent_dim, 
            output_shape,#-- frture --
            ):
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    decoder_outputs = {
        "c_bid":None,
        "c_ask":None,
        "c_volume":None,
        "p_bid":None,
        "p_ask":None,
        "p_volume":None
    }

    x = latent_inputs
    for units in dense_units:
        x = keras.layers.Dense(units, activation="tanh")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    for k in decoder_outputs.keys():
        decoder_outputs[k] = keras.layers.Dense(tf.math.reduce_prod(output_shape))(x)
        decoder_outputs[k] = keras.layers.Reshape(output_shape)(decoder_outputs[k])
        decoder_outputs[k] = keras.layers.ReLU(axis=1)(decoder_outputs[k])

    decoder = keras.Model(
        latent_inputs, outputs=[*decoder_outputs.values()], name="decoder"
    )

    return decoder