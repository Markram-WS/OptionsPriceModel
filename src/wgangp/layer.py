from tensorflow import keras

# import tensorflow as tf
# from   src.utils import debug


def discriminator(
    input_shape,
    dense_units,
    dropout_rate=0.2,
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    initializer=keras.initializers.GlorotUniform(),
    batch_size=32,
    activation=keras.layers.LeakyReLU(0.2),
):
    input_data = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = keras.layers.ZeroPadding2D((2, 2))(input_data)
    for unit in dense_units:
        x = keras.layers.Conv2D(
            unit,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            use_dropout=use_dropout,
            initializer=initializer,
        )(x)
        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        x = activation(x)
        if use_dropout:
            x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(1)(x)
    d_model = keras.models.Model(input_data, x, name="discriminator")
    return d_model


def generator(
    input_dim,
    dense_units,
    dropout_rate=0.2,
    use_bias=False,
    use_dropout=True,
    use_bn=True,
    initializer=keras.initializers.GlorotUniform(),
    batch_size=32,
    activation=keras.layers.LeakyReLU(0.2),
):
    input_data = keras.layers.Input(shape=(input_dim,), batch_size=batch_size)
    x = keras.layers.Dense(4 * 4 * 256, use_bias=False)(input_data)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    x = keras.layers.Reshape((4, 4, 256))(x)

    for unit in dense_units:
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(
            unit,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            use_dropout=use_dropout,
            initializer=initializer,
        )(x)

        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        if activation:
            x = activation(x)
        if use_dropout:
            x = keras.layers.Dropout(dropout_rate)(x)
    # At this point, we have an output which has the same shape as the input, (32, 32, 1).
    # We will use a Cropping2D layer to make it (28, 28, 1).
    x = keras.layers.Cropping2D((2, 2))(x)

    g_model = keras.models.Model(input_data, x, name="generator")
    return g_model
