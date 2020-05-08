import argparse
import errno
import os

import numpy as np
import horovod.tensorflow as hvd
import tensorflow as tf

import atom3d.psp.util as psp_util
import examples.cnn3d.model as model
import examples.cnn3d.feature_psp as feature

#tf.logging.set_verbosity(tf.logging.INFO)

# Training settings
parser = argparse.ArgumentParser(description='Tensorflow protein scoring prediction')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
args = parser.parse_args()


def conv_model(feature, target, is_training=True):
    per_res_scores, _, _, _ = model.scoring_model(
        model.NUM_RETYPE,
        feature,
        is_training=is_training,
        batch_norm=True,
        validation='softplus',
        final_activation='sigmoid')
    # Predict global score (i.e. average across local predictions for each
    # residue within a structure)
    predict = tf.reduce_mean(per_res_scores, keepdims=True, name='global_score')
    # Loss
    loss_op = tf.losses.mean_squared_error(target, predict)
    return predict, loss


def train_input_generator(x_train, y_train, batch_size=64):
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size


def main(_):
    # Horovod: initialize Horovod.
    hvd.init()

    # Download and load dataset.
    x_train, y_train = create_dataset(sharded, structures, scores_dir)

    # Build model...
    subgrid = tf.placeholder(
        tf.float32,
        [None, None, model.GRID_VOXELS * model.NB_TYPE],
        name='main_input')
    label = tf.placeholder(tf.float32, [None, 1], 'label')

    predict, loss = conv_model(subgrid, label, is_training=True)

    lr_scaler = hvd.size()
    # By default, Adasum doesn't need scaling when increasing batch size.
    # If used with NCCL, scale lr by local_size
    if args.use_adasum:
        lr_scaler = hvd.local_size() if hvd.nccl_built() else 1

    # Horovod: adjust learning rate based on lr_scaler.
    opt = tf.train.AdamOptimizer(0.001 * lr_scaler)

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(
        opt, op=hvd.Adasum if args.use_adasum else hvd.Average)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=20000 // hvd.size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                   every_n_iter=10),
    ]

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
    training_batch_generator = train_input_generator(
        x_train, y_train, batch_size=1)
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            subgrid_, label_ = next(training_batch_generator)
            mon_sess.run(train_op, feed_dict={subgrid: subgrid_, label: label_})


if __name__ == "__main__":
    tf.app.run()
