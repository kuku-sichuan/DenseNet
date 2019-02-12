from keras import backend as K
import tensorflow as tf

layers = tf.keras.layers


def conv(inputs, filters, kernel_size, data_format='channels_last'):
    return layers.Conv2D(filters, kernel_size,
                         padding='same',
                         data_format=data_format,
                         kernel_initializer="he_uniform",
                         use_bias=False)(inputs)


def Dense_Bottleneck(inputs, growth_rate, data_format='channels_first',
                                          bottleneck=True,
                                          dropout_rate=None,
                                          training=True):
    concat_axis = 1 if data_format == 'channels_first' else -1
    x = tf.layers.batch_normalization(inputs,
                                      axis=concat_axis,
                                      epsilon=1.1e-5,
                                      fused=True,
                                      training=training)
    x = layers.ReLU(max_value=6.0)(x)
    if bottleneck:
        inter_channel = growth_rate * 4
        x = conv(x, inter_channel, (1, 1), data_format)
        x = tf.layers.batch_normalization(inputs,
                                          axis=concat_axis,
                                          epsilon=1.1e-5,
                                          fused=True,
                                          training=training)
        x = layers.ReLU(max_value=6.0)(x)
    x = conv(x, growth_rate, (3, 3), data_format)
    if dropout_rate:
        x = tf.layers.dropout(x,
                              rate=dropout_rate,
                              training=training)
    return x


def DenseLayer(inputs, growth_rate, num_layers,
                                    data_format='channels_first',
                                    bottleneck=True,
                                    dropout_rate=None,
                                    training=True):
    concat_axis = 1 if data_format == 'channels_first' else -1
    x = inputs
    for i in range(num_layers):
        y = Dense_Bottleneck(x, growth_rate, data_format, bottleneck, dropout_rate, training)
        x = layers.concatenate([x, y], axis=concat_axis)
    return x


def TransLayer(inputs, data_format='channels_first',
                       compression=1.0,
                       dropout_rate=None,
                       training=True):
    concat_axis = 1 if data_format == 'channels_first' else -1
    output_channels = int(K.int_shape(inputs)[concat_axis]*compression)
    x = tf.layers.batch_normalization(inputs,
                                      axis=concat_axis,
                                      epsilon=1.1e-5,
                                      fused=True,
                                      training=training)
    x = layers.ReLU(max_value=6.0)(x)
    x = conv(x, output_channels, (1, 1), data_format)
    if dropout_rate:
        x = tf.layers.dropout(rate=dropout_rate,
                              training=training)(x)
    x = layers.AvgPool2D((2, 2),
                         strides=(2, 2),
                         data_format=data_format)(x)

    return x


def DenseNet(inputs, growth_rate, depth, num_dense_block,
                                         num_init_filters,
                                         sub_sample_image,
                                         num_classes,
                                         training=True,
                                         num_layer_list=-1,
                                         bottleneck=True,
                                         dropout_rate=None,
                                         compression=1.0,
                                         data_format='channels_first',
                                         all_config=None):

    if type(num_layer_list) is list or type(num_layer_list) is tuple:
        num_layers = list(num_layer_list)
        assert len(num_layers) == num_dense_block, "num_dense_block or num_layer_list is wrong"

        final_nb_layer = num_layers[-1]
        num_layers = num_layers[:-1]
    else:
        assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
        count = int((depth - 4) / 3)
        if bottleneck:
            count = count // 2

        num_layers = [count for _ in range(num_dense_block-1)]
        final_nb_layer = count

    if sub_sample_image:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)
    ########################################################################
    # Summary the image for inputs
    ########################################################################
    summary_inputs = inputs * tf.reshape(tf.convert_to_tensor(all_config.IMAGE_STD, tf.float32), (1, 3, 1, 1))
    summary_inputs += tf.reshape(tf.convert_to_tensor(all_config.IMAGE_MEANS, tf.float32), (1, 3, 1, 1))
    summary_inputs = tf.transpose(summary_inputs, [0, 2, 3, 1])
    tf.summary.image('inputs', summary_inputs, max_outputs=1, family='features')

    with tf.variable_scope("Stem_block"):
        x = layers.Conv2D(num_init_filters,
                          initial_kernel,
                          initial_strides,
                          padding='same',
                          data_format=data_format,
                          kernel_initializer='he_uniform',
                          name="a",
                          use_bias=False)(inputs)
        concat_axis = 1 if data_format == 'channels_first' else -1
        if sub_sample_image:
            x = tf.layers.batch_normalization(inputs,
                                              axis=concat_axis,
                                              epsilon=1.1e-5,
                                              fused=True,
                                              training=training)
            x = layers.ReLU(max_value=6.0)(x)
            x = layers.MaxPool2D((2, 2), (2, 2), data_format=data_format)(x)

    for block_idx in range(num_dense_block-1):
        with tf.variable_scope('DenseBlock_{}'.format(block_idx+1)):
            x = DenseLayer(x, growth_rate, num_layers[block_idx], data_format, bottleneck, dropout_rate, training)
        with tf.variable_scope('TransLayer_{}'.format(block_idx+1)):
            x = TransLayer(x, data_format, compression, dropout_rate, training)
    ##############################################################################
    # summary for last_layer images
    ##############################################################################
    summary_last_x = x[0]
    summary_last_x = tf.expand_dims(summary_last_x, axis=-1)
    tf.summary.image('last_features', summary_last_x, max_outputs=10, family='features')
    with tf.variable_scope('DenseBlock_{}'.format(num_dense_block)):
        x = DenseLayer(x, growth_rate, final_nb_layer, data_format, bottleneck, dropout_rate, training)
    with tf.variable_scope("Global_Avg"):
        x = tf.layers.batch_normalization(x, axis=concat_axis, epsilon=1.1e-5, fused=True, training=training)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.GlobalAveragePooling2D(data_format)(x)
    x = layers.Dense(num_classes, kernel_initializer='he_uniform')(x)

    return x


