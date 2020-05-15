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


def scoring_model(x, training, conv_drop_rate, fc_drop_rate, top_nn_drop_rate,
                  conv_filters, conv_kernel_size,
                  max_pool_positions,
                  max_pool_sizes, max_pool_strides,
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
    # FC 1.
    for layer, units in enumerate(fc_units):
        with tf.variable_scope("fc{:}".format(i)):
            x = _dense_batch_norm_dropout(x, units, training, batch_norm,
                                          dropout, fc_drop_rate, activation='relu')
            #variable_summaries(x)
            x = tf.Print(x, [tf.shape(x)],
                 message="FC layer {:}: ".format(layer),
                 first_n=1, summarize=10)

    with tf.variable_scope("out"):
        x = _dense_batch_norm_dropout(x, 1, training,
                                      batch_norm=False,
                                      dropout=False,
                                      drop_rate=top_nn_drop_rate,
                                      activation=None)
        #variable_summaries(x)

    with tf.variable_scope('logits'):
        logits = tf.reshape(x, shape=[-1], name='logits')
        #variable_summaries(logits)

    x = tf.Print(x, [tf.shape(x)],
         message="Output: ",
         first_n=1, summarize=10)

    return x


def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.999)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

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
