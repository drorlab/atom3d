from __future__ import division, print_function, absolute_import

import tensorflow as tf

try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass


def _dense_batch_norm_dropout(x, units, training, batch_norm,
                              dropout, drop_rate, activation='relu'):
    x = tf.layers.dense(
        x, units, activation=activation,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.zeros_initializer())
    return _batch_norm_dropout(x, training, batch_norm, dropout,
                               drop_rate=drop_rate)


def _batch_norm_dropout(x, training, batch_norm, dropout, drop_rate):
    if batch_norm:
        x = tf.layers.batch_normalization(
            x, fused=True, training=training, scale=True, momentum=0.99)
    if dropout:
        x = tf.layers.dropout(x, rate=drop_rate)
    return x


def base_network(x, training, conv_drop_rate, fc_drop_rate,
                 conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides,
                 fc_units,
                 batch_norm=True,
                 dropout=False):
    if batch_norm:
        x = tf.layers.batch_normalization(
            x, fused=True, training=training, scale=True, momentum=0.99)

    # Convs.
    for i in range(len(conv_filters)):
        with tf.variable_scope("conv{:d}".format(i)):
            x = tf.layers.conv3d(
                    x,
                    conv_filters[i],
                    kernel_size=conv_kernel_size,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    use_bias=True,
                    bias_initializer=tf.zeros_initializer(),
                    activation='relu',
                    padding='VALID')

            if max_pool_positions[i]:
                x = tf.layers.max_pooling3d(
                        x,
                        pool_size=max_pool_sizes[i],
                        strides=max_pool_strides[i],
                        padding="SAME",
                        name="fpool{:}".format(i))
            x = _batch_norm_dropout(x, training, batch_norm, dropout,
                                    drop_rate=conv_drop_rate)
            #variable_summaries(x)
            x = tf.Print(x, [tf.shape(x)],
                 message="Conv layer {:}: ".format(i),
                 first_n=1, summarize=10)

    x = tf.layers.flatten(x)
    # FC layers.
    for layer, units in enumerate(fc_units):
        with tf.variable_scope("fc{:}".format(i)):
            x = _dense_batch_norm_dropout(x, units, training, batch_norm,
                                          dropout, fc_drop_rate, activation='relu')
            #variable_summaries(x)
            x = tf.Print(x, [tf.shape(x)],
                 message="FC layer {:}: ".format(layer),
                 first_n=1, summarize=10)

    x = tf.identity(x, name='fcfinal')
    return x


def single_model(x, training, conv_drop_rate, fc_drop_rate, top_nn_drop_rate,
                 conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides,
                 fc_units,
                 batch_norm=True,
                 dropout=False,
                 top_nn_activation=None):

    x = base_network(x, training, conv_drop_rate, fc_drop_rate,
                     conv_filters, conv_kernel_size,
                     max_pool_positions, max_pool_sizes, max_pool_strides,
                     fc_units,
                     batch_norm,
                     dropout)

    with tf.variable_scope("out"):
        x = _dense_batch_norm_dropout(x, 1, training,
                                      batch_norm=False,
                                      dropout=False,
                                      drop_rate=top_nn_drop_rate,
                                      activation=top_nn_activation)
        #variable_summaries(x)

    with tf.variable_scope('logits'):
        logits = tf.identity(x, name='logits')
        #variable_summaries(logits)

    logits = tf.Print(logits, [tf.shape(logits)],
         message="Output: ",
         first_n=1, summarize=10)

    return logits


def siamese_model(x, training, conv_drop_rate, fc_drop_rate, top_nn_drop_rate,
                 conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides,
                 fc_units,
                 top_fc_units,
                 batch_norm=True,
                 dropout=False,
                 top_nn_activation=None):

    grid_left = x[:, 0]
    grid_right = x[:, 1]

    with tf.variable_scope('base_networks', reuse=tf.AUTO_REUSE):
        processed_left = base_network(
            grid_left, training, conv_drop_rate, fc_drop_rate,
            conv_filters, conv_kernel_size,
            max_pool_positions, max_pool_sizes, max_pool_strides,
            fc_units,
            batch_norm, dropout)

    with tf.variable_scope('base_networks', reuse=tf.AUTO_REUSE):
        processed_right = base_network(
            grid_right, training, conv_drop_rate, fc_drop_rate,
            conv_filters, conv_kernel_size,
            max_pool_positions, max_pool_sizes, max_pool_strides,
            fc_units,
            batch_norm, dropout)

    x = tf.concat([processed_left, processed_right], 1, name='concat')

    # Deep non-siamese.
    with tf.variable_scope("top_nn"):
        for layer, units in enumerate(top_fc_units):
            with tf.variable_scope("fc{:d}".format(layer)):
                x = _dense_batch_norm_dropout(x, units, training, batch_norm,
                                              dropout, top_nn_drop_rate,
                                              activation='relu')
        #variable_summaries(x)

    x = tf.identity(x, name='embedding')

    with tf.variable_scope("out"):
        x = _dense_batch_norm_dropout(x, 1, training,
                                      batch_norm=False,
                                      dropout=False,
                                      drop_rate=top_nn_drop_rate,
                                      activation=top_nn_activation)
        #variable_summaries(x)

    with tf.variable_scope('logits'):
        logits = tf.identity(x, name='logits')
        #variable_summaries(logits)

    logits = tf.Print(logits, [tf.shape(logits)],
         message="Output: ",
         first_n=1, summarize=10)

    return logits


def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.999)

    # Update moving_mean and moving_variance of batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    tf_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step=tf_step)
    train_op = tf.group([train_op, update_ops], name='train_op')

    return train_op


def variable_summaries(var):
    """Attach summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        #tf.summary.histogram('histogram', var)
