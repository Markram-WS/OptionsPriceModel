from tensorflow import keras


def generatorDense(
    input_dim,
    output_dim,
    dense_units,
    dropout_rate=0.2,
    use_bias=False,
    use_dropout=True,
    use_bn=True,
    initializer=keras.initializers.GlorotUniform(),
    batch_size=32,
    activation=keras.layers.LeakyReLU(0.2),
):
    input_data = keras.layers.Input(
        shape=input_dim, batch_size=batch_size, name="input_data"
    )
    x = keras.layers.Flatten()(input_data)
    x = keras.layers.BatchNormalization()(x)
    x = activation(x)
    for layer_i, unit in enumerate(dense_units):
        x = keras.layers.Dense(
            unit,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=initializer,
            name=f"Dense{layer_i}_{unit}",
        )(x)
        if use_bn:
            x = keras.layers.BatchNormalization(name=f"BatchNorm{layer_i}_{unit}")(x)
        if use_dropout:
            x = keras.layers.Dropout(dropout_rate, name=f"Dropout{layer_i}_{unit}")(x)

    x = keras.layers.Dense(
        output_dim[0] * output_dim[1],
    )(x)
    x = keras.layers.Reshape(output_dim, name="Reshape-output")(x)
    g_model = keras.models.Model(input_data, x, name="generator")
    return g_model


def generatorConv2D(
    input_dim,
    output_dim,
    dense_units,
    dropout_rate=0.2,
    use_bias=False,
    use_dropout=True,
    use_bn=True,
    initializer=keras.initializers.GlorotUniform(),
    batch_size=32,
    activation=keras.layers.LeakyReLU(0.2),
):
    input_data = keras.layers.Input(
        shape=input_dim, batch_size=batch_size, name="input_data"
    )
    x = keras.layers.Flatten()(input_data)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = keras.layers.Reshape((input_dim[-1], input_dim[-1], input_dim[-1]))(input_data)
    for layer_i, unit in enumerate(dense_units):
        # x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2DTranspose(
            unit,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            name=f"Conv2D{layer_i}-{unit}",
        )(x)

        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        if activation:
            x = activation(x)
        if use_dropout:
            x = keras.layers.Dropout(dropout_rate)(x)

    # Convolutional layer to adjust the number of output channels to 4
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(
        output_dim[0] * output_dim[1],
    )(x)
    x = keras.layers.Reshape(output_dim, name="Reshape-output")(x)
    g_model = keras.models.Model(input_data, x, name="generator")
    return g_model
