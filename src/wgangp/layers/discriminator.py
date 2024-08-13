from tensorflow import keras

import tensorflow as tf


def discriminatorDense(
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
    input_data = keras.layers.Input(
        shape=input_shape, batch_size=batch_size, name="input_y_data"
    )
    x = keras.layers.Flatten()(input_data)
    x = keras.layers.BatchNormalization()(x)
    x = activation(x)
    for unit in dense_units:
        x = keras.layers.Dense(
            unit,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=initializer,
            name=f"Dense_{unit}",
        )(x)
        if use_bn:
            x = keras.layers.BatchNormalization(name=f"BatchNorm_{unit}")(x)
        if use_dropout:
            x = keras.layers.Dropout(dropout_rate, name=f"Dropout_{unit}")(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(
        1,
        name="logic",
    )(x)
    d_model = keras.models.Model(input_data, x, name="discriminator")
    return d_model


def discriminatorConv2D(
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
    input_data = keras.layers.Input(
        shape=input_shape, batch_size=batch_size, name="input_y_data"
    )
    input_data_expand = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(
        input_data
    )

    x = keras.layers.ZeroPadding2D((2, 2))(input_data_expand)
    for unit in dense_units:
        x = keras.layers.Conv2D(
            unit,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
        )(x)
        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        x = activation(x)
        if use_dropout:
            x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(
        1,
        name="Conv2D-logic",
    )(x)
    d_model = keras.models.Model(input_data, x, name="discriminator")
    return d_model
