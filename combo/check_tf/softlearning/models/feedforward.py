import tensorflow as tf


from softlearning.utils.keras import PicklableKerasModel


def feedforward_model(input_shapes,
                      output_size,
                      hidden_layer_sizes,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      # kernel_initializer='glorot_uniform',
                      kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'),
                      name='feedforward_model',
                      *args,
                      **kwargs):
    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    if preprocessors is None:
        preprocessors = (None, ) * len(inputs)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_ in zip(preprocessors, inputs)
    ]

    concatenated = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(preprocessed_inputs)

    out = concatenated
    for units in hidden_layer_sizes:
        out = tf.keras.layers.Dense(
            units, *args, activation=activation, kernel_initializer=kernel_initializer,
            bias_initializer=tf.keras.initializers.Constant(value=0.1), **kwargs
        )(out)

    out = tf.keras.layers.Dense(
        output_size, *args, activation=output_activation,
        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
        bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
        **kwargs
    )(out)

    model = PicklableKerasModel(inputs, out, name=name)

    return model
