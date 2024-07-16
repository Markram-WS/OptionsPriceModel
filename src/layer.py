from tensorflow import keras
import tensorflow as tf
from   src.utils import debug


def encoder(
    latent_dim, 
    input_shape, 
    dense_units, 
    dropout_rate=0.1,
    use_bias = False,
    initializer = keras.initializers.GlorotUniform(),
    batch_size=32
):    
    features_input = keras.layers.Input(shape=input_shape,batch_size=batch_size , name='features') 
    x = keras.layers.Reshape((32, 2, 1))(features_input)  # Reshape to (32, 2, 1) for Conv2D

    # Propagate through one or more densely connected layers
    for units in dense_units:
        #<NN>
        # x = keras.layers.Dense( units, 
        #                         activation="relu",
        #                         kernel_initializer=initializer,
        #                         use_bias=use_bias
        #                         )(x)
        # x = keras.layers.Dropout(dropout_rate)(x)
        
        #<CNN>
        x = keras.layers.Conv2D(units, (3, 3), activation='relu', padding='same',kernel_initializer=initializer)(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Flatten()(x)

    z_mean = keras.layers.Dense(
            latent_dim, 
            kernel_initializer=initializer,
            use_bias=use_bias,
            dtype="float32", 
            name="z_mean")(x)
    log_var = keras.layers.Dense(
            latent_dim, 
            kernel_initializer=initializer,
            use_bias=use_bias,
            dtype="float32", 
            name="log_var")(x)

    encoder = keras.Model([features_input], [z_mean, log_var], name="encoder")

    return encoder

def decoder(latent_dim, 
            output_shape, 
            dense_units, 
            dropout_rate=0.1 ,   
            use_bias = False,
            initializer = keras.initializers.GlorotUniform()):
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = latent_inputs
    # สร้าง dense layer ตาม latent_dim
    x = keras.layers.Dense(6 * latent_dim, activation='relu')(x)

    # แปลง vector เป็นรูปแบบ 3D tensor
    x = keras.layers.Reshape((6, 32, 1))(x)

    for units in dense_units:
        ##<NN>
        # x = keras.layers.Dense(units, 
        #         activation="relu",
        #         kernel_initializer=initializer,
        #         use_bias=use_bias,
        #         dtype="float32"
        #         )(x)
        # x = keras.layers.Dropout(dropout_rate)(x)
        
        ##<CNN>
        # สร้าง CNN layers
        x = keras.layers.Conv2DTranspose(units, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2DTranspose(4, (3, 3), activation='relu', padding='same')(x)
    # สร้าง output layer โดยกำหนดให้มี shape (None, 6, 32)
    decoder_outputs = keras.layers.Reshape((32, 4))(x)  # Reshape to (32, 4) to match output shape(x)
    # decoder_outputs = {
    #     "c_bid": None,
    #     "c_ask": None,
    #     "c_volume": None,
    #     "p_bid": None,
    #     "p_ask": None,
    #     "p_volume": None
    # }
    # for k in decoder_outputs.keys():
    #     decoder_outputs[k] = keras.layers.Dense(output_shape[0] * output_shape[1], name=f"Dense_{k}")(x)
    #     decoder_outputs[k] = keras.layers.Reshape(output_shape, name=f"Reshape_{k}")(decoder_outputs[k])
    #     decoder_outputs[k] = keras.layers.ReLU(name=f"ReLU_{k}")(decoder_outputs[k])

    decoder = keras.Model(
        latent_inputs, outputs=decoder_outputs, name="decoder"
    )

    return decoder

