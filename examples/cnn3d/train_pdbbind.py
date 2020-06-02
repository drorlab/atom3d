from __future__ import division, print_function, absolute_import

import argparse
import functools
import json
import logging
import math
import os
import random
import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

import atom3d.util.shard as sh
import examples.cnn3d.model as model
import examples.cnn3d.feature_pdbbind as feature_pdbbind
import examples.cnn3d.subgrid_gen as subgrid_gen


def compute_stats(results):
    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')
    # Compute RMSE
    res['rmse'] = ((all_true - all_pred)**2).mean()**0.5
    return res


# Construct model and loss
def conv_model(feature, target, is_training, conv_drop_rate, fc_drop_rate,
               top_nn_drop_rate, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2**n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1]*int((num_conv+1)/2)
    max_pool_sizes = [2]*num_conv
    max_pool_strides = [2]*num_conv
    fc_units = [512]

    output = model.single_model(
        feature,
        is_training,
        conv_drop_rate,
        fc_drop_rate,
        top_nn_drop_rate,
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        batch_norm=args.use_batch_norm,
        dropout=not args.no_dropout,
        top_nn_activation=args.top_nn_activation)

    # Prediction
    predict = tf.identity(output, name='predict')
    # Loss
    loss = tf.losses.mean_squared_error(target, predict)
    return predict, loss


def batch_dataset_generator(gen, args, is_testing=False):
    grid_size = subgrid_gen.grid_size(args.grid_config)
    channel_size = subgrid_gen.num_channels(args.grid_config)
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.string, tf.float32, tf.float32),
        output_shapes=((), (grid_size, grid_size, grid_size, channel_size), (1,))
        )

    # Shuffle dataset
    if not is_testing:
        if args.shuffle:
            dataset = dataset.repeat(count=None)
        else:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=1000))

    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(8)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return dataset, next_element


def train_model(sess, args):
    # tf Graph input
    # Subgrid maps for each residue in a protein
    logging.debug('Create input placeholder...')
    grid_size = subgrid_gen.grid_size(args.grid_config)
    channel_size = subgrid_gen.num_channels(args.grid_config)
    feature_placeholder = tf.placeholder(
        tf.float32,
        [None, grid_size, grid_size, grid_size, channel_size],
        name='main_input')
    label_placeholder = tf.placeholder(tf.float32, [None, 1], 'label')

    # Placeholder for model parameters
    training_placeholder = tf.placeholder(tf.bool, shape=[], name='is_training')
    conv_drop_rate_placeholder = tf.placeholder(tf.float32, name='conv_drop_rate')
    fc_drop_rate_placeholder = tf.placeholder(tf.float32, name='fc_drop_rate')
    top_nn_drop_rate_placeholder = tf.placeholder(tf.float32, name='top_nn_drop_rate')

    # Define loss and optimizer
    logging.debug('Define loss and optimizer...')
    predict_op, loss_op = conv_model(
        feature_placeholder, label_placeholder, training_placeholder,
        conv_drop_rate_placeholder, fc_drop_rate_placeholder,
        top_nn_drop_rate_placeholder, args)
    logging.debug('Generate training ops...')
    train_op = model.training(loss_op, args.learning_rate)

    # Initialize the variables (i.e. assign their default value)
    logging.debug('Initializing global variables...')
    init = tf.global_variables_initializer()

    # Create saver and summaries.
    logging.debug('Initializing saver...')
    saver = tf.train.Saver(max_to_keep=100000)
    logging.debug('Finished initializing saver...')

    def __loop(generator, mode, num_iters):
        tf_dataset, next_element = batch_dataset_generator(
            generator, args, is_testing=(mode=='test'))

        structs, losses, preds, labels = [], [], [], []
        epoch_loss = 0
        progress_format = mode + ' loss: {:6.6f}'

        # Loop over all batches (one batch is all feature for 1 protein)
        num_batches = int(math.ceil(float(num_iters)/args.batch_size))
        #print('Running {:} -> {:} iters in {:} batches (batch size: {:})'.format(
        #    mode, num_iters, num_batches, args.batch_size))
        with tqdm.tqdm(total=num_batches, desc=progress_format.format(0)) as t:
            for i in range(num_batches):
                try:
                    struct_, feature_, label_ = sess.run(next_element)
                    _, pred, loss = sess.run(
                        [train_op, predict_op, loss_op],
                        feed_dict={feature_placeholder: feature_,
                                   label_placeholder: label_,
                                   training_placeholder: (mode == 'train'),
                                   conv_drop_rate_placeholder:
                                       args.conv_drop_rate if mode == 'train' else 0.0,
                                   fc_drop_rate_placeholder:
                                       args.fc_drop_rate if mode == 'train' else 0.0,
                                   top_nn_drop_rate_placeholder:
                                       args.top_nn_drop_rate if mode == 'train' else 0.0})
                    epoch_loss += (np.mean(loss) - epoch_loss) / (i + 1)
                    structs.extend(struct_)
                    losses.append(loss)
                    preds.extend(pred)
                    labels.extend(label_)

                    t.set_description(progress_format.format(epoch_loss))
                    t.update(1)
                except StopIteration:
                    logging.info("\nEnd of dataset at iteration {:}".format(i))
                    break

        def __concatenate(array):
            try:
                array = np.concatenate(array)
                return array
            except:
                return array

        structs = __concatenate(structs)
        preds = __concatenate(preds)
        labels = __concatenate(labels)
        losses = __concatenate(losses)
        return structs, preds, labels, losses, epoch_loss

    # Run the initializer
    logging.debug('Running initializer...')
    sess.run(init)
    logging.debug('Finished running initializer...')

    ##### Training + validation
    prev_val_loss, best_val_loss = float("inf"), float("inf")

    if (args.max_pdbs_train == None):
        pdbcodes = feature_pdbbind.read_split(args.train_split_filename)
        train_num_structs = len(pdbcodes)
    else:
        train_num_structs = args.max_pdbs_train

    if (args.max_pdbs_val == None):
        pdbcodes = feature_pdbbind.read_split(args.val_split_filename)
        val_num_structs = len(pdbcodes)
    else:
        val_num_structs = args.max_pdbs_val

    train_num_structs *= args.repeat_gen
    val_num_structs *= args.repeat_gen

    logging.info("Start training with {:} structs for train and {:} structs for val per epoch".format(
        train_num_structs, val_num_structs))


    def _save():
        ckpt = saver.save(sess, os.path.join(args.output_dir, 'model-ckpt'),
                          global_step=epoch)
        return ckpt

    run_info_filename = os.path.join(args.output_dir, 'run_info.json')
    run_info = {}
    def __update_and_write_run_info(key, val):
        run_info[key] = val
        with open(run_info_filename, 'w') as f:
            json.dump(run_info, f, indent=4)

    per_epoch_val_losses = []
    for epoch in range(1, args.num_epochs+1):
        random_seed = args.random_seed #random.randint(1, 10e6)
        logging.info('Epoch {:} - random_seed: {:}'.format(epoch, args.random_seed))

        logging.debug('Creating train generator...')
        train_generator_callable = functools.partial(
            feature_pdbbind.dataset_generator,
            args.data_filename,
            args.train_split_filename,
            args.labels_filename,
            args.grid_config,
            shuffle=args.shuffle,
            repeat=args.repeat_gen,
            max_pdbs=args.max_pdbs_train,
            random_seed=random_seed)

        logging.debug('Creating val generator...')
        val_generator_callable = functools.partial(
            feature_pdbbind.dataset_generator,
            args.data_filename,
            args.val_split_filename,
            args.labels_filename,
            args.grid_config,
            shuffle=args.shuffle,
            repeat=args.repeat_gen,
            max_pdbs=args.max_pdbs_val,
            random_seed=random_seed)

        # Training
        train_structs, train_preds, train_labels, _, curr_train_loss = __loop(
            train_generator_callable, 'train', num_iters=train_num_structs)
        # Validation
        val_structs, val_preds, val_labels, _, curr_val_loss = __loop(
            val_generator_callable, 'val', num_iters=val_num_structs)

        per_epoch_val_losses.append(curr_val_loss)
        __update_and_write_run_info('val_losses', per_epoch_val_losses)

        if args.use_best or args.early_stopping:
            if curr_val_loss < best_val_loss:
                # Found new best epoch.
                best_val_loss = curr_val_loss
                ckpt = _save()
                __update_and_write_run_info('val_best_loss', best_val_loss)
                __update_and_write_run_info('best_ckpt', ckpt)
                logging.info("New best {:}".format(ckpt))

        if (epoch == args.num_epochs - 1 and not args.use_best):
            # At end and just using final checkpoint.
            ckpt = _save()
            __update_and_write_run_info('best_ckpt', ckpt)
            logging.info("Last checkpoint {:}".format(ckpt))

        if args.save_all_ckpts:
            # Save at every checkpoint
            ckpt = _save()
            logging.info("Saving checkpoint {:}".format(ckpt))

        if args.early_stopping and curr_val_loss >= prev_val_loss:
            logging.info("Validation loss stopped decreasing, stopping...")
            break
        else:
            prev_val_loss = curr_val_loss

    logging.info("Finished training")

    ## Save last train and val results
    logging.info("Saving train and val results")
    train_df = pd.DataFrame(
        np.array([train_structs, train_labels, train_preds]).T,
        columns=['structure', 'true', 'pred'],
        )
    train_df.to_pickle(os.path.join(args.output_dir, 'train_result.pkl'))

    val_df = pd.DataFrame(
        np.array([val_structs, val_labels, val_preds]).T,
        columns=['structure', 'true', 'pred'],
        )
    val_df.to_pickle(os.path.join(args.output_dir, 'val_result.pkl'))


    ##### Testing
    to_use = run_info['best_ckpt'] if args.use_best else ckpt
    logging.info("Using {:} for testing".format(to_use))
    saver.restore(sess, to_use)

    test_generator_callable = functools.partial(
        feature_pdbbind.dataset_generator,
        args.data_filename,
        args.test_split_filename,
        args.labels_filename,
        args.grid_config,
        shuffle=args.shuffle,
        repeat=1,
        max_pdbs=args.max_pdbs_test,
        random_seed=args.random_seed)

    if (args.max_pdbs_test == None):
        pdbcodes = feature_pdbbind.read_split(args.test_split_filename)
        test_num_structs = len(pdbcodes)
    else:
        test_num_structs = args.max_pdbs_test

    logging.info("Start testing with {:} structs".format(test_num_structs))


    test_structs, test_preds, test_labels, _, test_loss = __loop(
        test_generator_callable, 'test', num_iters=test_num_structs)
    logging.info("Finished testing")

    test_df = pd.DataFrame(
        np.array([test_structs, test_labels, test_preds]).T,
        columns=['structure', 'true', 'pred'],
        )
    test_df.to_pickle(os.path.join(args.output_dir, 'test_result.pkl'))

    # Compute global correlations
    res = compute_stats(test_df)
    logging.info(
        '\nStats\n'
        '    RMSE: {:.3f}\n'
        '    Pearson: {:.3f}\n'
        '    Spearman: {:.3f}'.format(
        float(res["rmse"]),
        float(res["all_pearson"]),
        float(res["all_spearman"])))


def create_train_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_filename', type=str,
        default='/home/users/psuriana/atom3d/examples/cnn3d/data/pdbbind/pdbbind_3dcnn.h5')
    parser.add_argument(
        '--labels_filename', type=str,
        default='/home/users/psuriana/atom3d/examples/cnn3d/data/pdbbind/pdbbind_refined_set_labels.csv')

    parser.add_argument(
        '--train_split_filename', type=str,
        default='/home/users/psuriana/atom3d/examples/cnn3d/data/pdbbind/splits/split_identity60/train_identity60.txt')
    parser.add_argument(
        '--val_split_filename', type=str,
        default='/home/users/psuriana/atom3d/examples/cnn3d/data/pdbbind/splits/split_identity60/val_identity60.txt')
    parser.add_argument(
        '--test_split_filename', type=str,
        default='/home/users/psuriana/atom3d/examples/cnn3d/data/pdbbind/splits/split_identity60/test_identity60.txt')

    parser.add_argument(
        '--output_dir', type=str,
        default='/scratch/users/psuriana/atom3d/model')

    # Training parameters
    parser.add_argument('--max_pdbs_train', type=int, default=None)
    parser.add_argument('--max_pdbs_val', type=int, default=None)
    parser.add_argument('--max_pdbs_test', type=int, default=None)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)
    parser.add_argument('--top_nn_drop_rate', type=float, default=0.5)
    parser.add_argument('--top_nn_activation', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--repeat_gen', type=int, default=200)

    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--use_batch_norm', action='store_true', default=False)
    parser.add_argument('--no_dropout', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--use_best', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=random.randint(1, 10e6))
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--unobserved', action='store_true', default=False)
    parser.add_argument('--save_all_ckpts', action='store_true', default=False)

    return parser


def main():
    parser = create_train_parser()
    args = parser.parse_args()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.info("Running 3D CNN PDB-Bind training...")

    if args.unobserved:
        args.output_dir = os.path.join(args.output_dir, 'None')
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        num = 0
        while True:
            dirpath = os.path.join(args.output_dir, str(num))
            if os.path.exists(dirpath):
                num += 1
            else:
                args.output_dir = dirpath
                logging.info('Creating output directory {:}'.format(args.output_dir))
                os.mkdir(args.output_dir)
                break

    args.__dict__['grid_config'] = feature_pdbbind.grid_config
    logging.info("\n" + str(json.dumps(args.__dict__, indent=4)) + "\n")

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    logging.info("Writing all output to {:}".format(args.output_dir))
    with tf.Session() as sess:
        np.random.seed(args.random_seed)
        tf.set_random_seed(args.random_seed)
        train_model(sess, args)


if __name__ == '__main__':
    main()
