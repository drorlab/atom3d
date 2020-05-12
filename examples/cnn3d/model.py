from __future__ import division, print_function, absolute_import

import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

NUM_RETYPE = 15

GRID_SIZE = 24
GRID_VOXELS = GRID_SIZE * GRID_SIZE * GRID_SIZE
NB_TYPE = 169


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32,
                           tf.truncated_normal_initializer(stddev=0.01))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32,
                           tf.constant_initializer(0.1, dtype=tf.float32))


def scoring_model(num_retype, maps, is_training, batch_norm=True,
                  validation='softplus', final_activation='sigmoid'):

    prev_layer = maps

    retyper = _weight_variable("retype"+"_"+str(num_retype), [NB_TYPE, num_retype])
    with tf.name_scope('Retype'):
        tf.summary.histogram("Weights_R", retyper)
    map_shape = tf.gather(tf.shape(prev_layer), [0, 1, 2, 3]) # Extract the first three dimensions

    map_shape = tf.concat([map_shape, [num_retype]], axis=0)
    prev_layer = tf.reshape(prev_layer, [-1, NB_TYPE])
    prev_layer = tf.matmul(prev_layer, retyper);
    retyped = tf.reshape(prev_layer, map_shape)


    CONV1_OUT = 20

    kernelConv1 = _weight_variable("weights_C1"+"_"+str(num_retype),
                                   [3, 3, 3, num_retype, CONV1_OUT])
    prev_layer = tf.nn.conv3d(retyped, kernelConv1, [1, 1, 1, 1, 1],
                              padding='VALID')
    biasConv1 = _bias_variable("biases_C1"+"_"+str(num_retype), [CONV1_OUT])

    with tf.name_scope('Conv1'):
        tf.summary.histogram("weights_C1", kernelConv1)
        tf.summary.histogram("bias_C1", biasConv1)

    prev_layer = prev_layer + biasConv1;

    if batch_norm :
        prev_layer = tf.layers.batch_normalization(
            prev_layer, training=is_training, name="batchn1")

    prev_layer = tf.nn.dropout(
        prev_layer, 1 -  tf.cast(is_training, dtype=tf.float32) * 0.5,
        name="dropout1")
    if validation == 'softplus':
        conv1 = tf.nn.softplus(prev_layer, name="softplus1")
    else:
        conv1 = tf.nn.elu(prev_layer, name="elu1")


    CONV2_OUT = 30

    kernelConv2 = _weight_variable("weights_C2"+"_"+str(num_retype),
                                   [4, 4, 4, CONV1_OUT, CONV2_OUT])
    prev_layer = tf.nn.conv3d(conv1, kernelConv2, [1, 1, 1, 1, 1],
                              padding='VALID')
    biasConv2 = _bias_variable("biases_C2"+"_"+str(num_retype), [CONV2_OUT])

    with tf.name_scope('Conv2'):
        tf.summary.histogram("weights_C2", kernelConv2)
        tf.summary.histogram("bias_C2", biasConv2)

    prev_layer = prev_layer + biasConv2;
    if batch_norm :
        prev_layer = tf.layers.batch_normalization(
            prev_layer, training=is_training, name="batchn2")

    if validation == 'softplus':
        prev_layer = tf.nn.softplus(prev_layer, name="softplus2")
    else:
        prev_layer = tf.nn.elu(prev_layer, name="elu2")


    CONV3_OUT = 20

    kernelConv3 = _weight_variable("weights_C3"+"_"+str(num_retype),
                                   [4, 4, 4, CONV2_OUT, CONV3_OUT])
    prev_layer = tf.nn.conv3d(prev_layer, kernelConv3, [1, 1, 1, 1, 1],
                              padding='VALID')
    biasConv3 = _bias_variable("biases_C3"+"_"+str(num_retype), [CONV3_OUT])

    with tf.name_scope('Conv3'):
        tf.summary.histogram("weights_C3", kernelConv3)
        tf.summary.histogram("bias_C3", biasConv3)

    prev_layer = prev_layer + biasConv3;

    if batch_norm :
        prev_layer = tf.layers.batch_normalization(
            prev_layer, training=is_training, name="batchn3")

    if validation == 'softplus':
        prev_layer = tf.nn.softplus(prev_layer, name="softplus3")
    else:
        prev_layer = tf.nn.elu(prev_layer, name="elu3")


    POOL_SIZE = 4
    prev_layer = tf.nn.avg_pool3d(
        prev_layer,
        [1, POOL_SIZE, POOL_SIZE, POOL_SIZE, 1],
        [1, POOL_SIZE, POOL_SIZE, POOL_SIZE, 1],
        padding='VALID')

    NB_DIMOUT = 4*4*4*CONV3_OUT
    flat0 = tf.reshape(prev_layer,[-1,NB_DIMOUT])

    LINEAR1_OUT = 160

    weightsLinear = _weight_variable("weights_L1"+"_"+str(num_retype),
                                     [NB_DIMOUT, LINEAR1_OUT])
    prev_layer = tf.matmul(flat0, weightsLinear)
    biasLinear1 = _bias_variable("biases_L1"+"_"+str(num_retype), [LINEAR1_OUT])

    with tf.name_scope('Linear1'):
        tf.summary.histogram("weights_L1", weightsLinear)
        tf.summary.histogram("biases_L1", biasLinear1)

    prev_layer = prev_layer + biasLinear1

    if batch_norm:
        prev_layer = tf.layers.batch_normalization(
            prev_layer, training=is_training, name="batchn4")

    if validation == 'softplus':
        flat1 = tf.nn.softplus(prev_layer, name="softplus3")
    else:
        flat1 = tf.nn.elu(prev_layer, name="elu1")


    weightsLinear2 = _weight_variable("weights_L2"+"_"+str(num_retype),
                                      [LINEAR1_OUT, 1])

    with tf.name_scope('Linear2'):
        tf.summary.histogram("weights_L2", weightsLinear2)

    last = tf.matmul(flat1, weightsLinear2)
    prev_layer = tf.squeeze(last)

    if final_activation == 'tanh':
        return (tf.add(tf.tanh(prev_layer)*0.5, 0.5, name = "main_output"),
                flat1, last, weightsLinear2)
    else:
        return (tf.sigmoid(prev_layer, name = "main_output"),
                flat1, last, weightsLinear2)


def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.999)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
