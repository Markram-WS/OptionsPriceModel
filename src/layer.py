from tensorflow import keras
import tensorflow as tf
from   src.utils import debug


def encoder(
    latent_dim, 
    input_shape, 
    dense_units, 
    dropout_rate=0.1,
    use_bias = False,
    batch_size=32
):    
    features_input = keras.layers.Input(shape=input_shape,batch_size=batch_size , name='features') 
    
    # Reduce 2-D representation of molecule to 1-D
    x = keras.layers.GlobalAveragePooling1D()(features_input)

    # Propagate through one or more densely connected layers
    for units in dense_units:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    z_mean = keras.layers.Dense(latent_dim, dtype="float32", name="z_mean")(x)
    log_var = keras.layers.Dense(latent_dim, dtype="float32", name="log_var")(x)

    encoder = keras.Model([features_input], [z_mean, log_var], name="encoder")

    return encoder

def decoder(latent_dim, output_shape, dense_units, dropout_rate=0.1):
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    decoder_outputs = {
        "c_bid": None,
        "c_ask": None,
        "c_volume": None,
        "p_bid": None,
        "p_ask": None,
        "p_volume": None
    }

    x = latent_inputs
    for units in dense_units:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    for k in decoder_outputs.keys():
        decoder_outputs[k] = keras.layers.Dense(output_shape[0] * output_shape[1], name=f"Dense_{k}")(x)
        decoder_outputs[k] = keras.layers.Reshape(output_shape, name=f"Reshape_{k}")(decoder_outputs[k])
        decoder_outputs[k] = keras.layers.ReLU(name=f"ReLU_{k}")(decoder_outputs[k])

    decoder = keras.Model(
        latent_inputs, outputs=list(decoder_outputs.values()), name="decoder"
    )

    return decoder

