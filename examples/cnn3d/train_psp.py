from __future__ import division, print_function, absolute_import

import argparse
import functools
import json
import logging
import math
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

import atom3d.shard.shard as sh
import atom3d.psp.util as psp_util
import examples.cnn3d.model as model
import examples.cnn3d.feature_psp as feature_psp
import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util

import dotenv as de
de.load_dotenv(de.find_dotenv(usecwd=True))


def compute_global_correlations(results):
    per_target = []
    for key, val in results.groupby(['target']):
        # Ignore target with 2 decoys only since the correlations are
        # not really meaningful.
        if val.shape[0] < 3:
            continue
        true = val['true'].astype(float)
        pred = val['pred'].astype(float)
        pearson = true.corr(pred, method='pearson')
        kendall = true.corr(pred, method='kendall')
        spearman = true.corr(pred, method='spearman')
        per_target.append((key, pearson, kendall, spearman))
    per_target = pd.DataFrame(
        data=per_target,
        columns=['target', 'pearson', 'kendall', 'spearman'])

    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')

    res['per_target_mean_pearson'] = per_target['pearson'].mean()
    res['per_target_mean_kendall'] = per_target['kendall'].mean()
    res['per_target_mean_spearman'] = per_target['spearman'].mean()

    res['per_target_median_pearson'] = per_target['pearson'].median()
    res['per_target_median_kendall'] = per_target['kendall'].median()
    res['per_target_median_spearman'] = per_target['spearman'].median()
    return res


def __stats(mode, df):
    # Compute global correlations
    res = compute_global_correlations(df)
    logging.info(
        '\n{:}\n'
        'Correlations (Pearson, Kendall, Spearman)\n'
        '    per-target averaged median: ({:.3f}, {:.3f}, {:.3f})\n'
        '    per-target averaged mean: ({:.3f}, {:.3f}, {:.3f})\n'
        '    all averaged: ({:.3f}, {:.3f}, {:.3f})'.format(
        mode,
        float(res["per_target_median_pearson"]),
        float(res["per_target_median_kendall"]),
        float(res["per_target_median_spearman"]),
        float(res["per_target_mean_pearson"]),
        float(res["per_target_mean_kendall"]),
        float(res["per_target_mean_spearman"]),
        float(res["all_pearson"]),
        float(res["all_kendall"]),
        float(res["all_spearman"])))


# Construct model and loss
def conv_model(feature, target, is_training, conv_drop_rate, fc_drop_rate,
               top_nn_drop_rate, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2**n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1]*int((num_conv+1)/2)
    max_pool_sizes = [4]*num_conv
    max_pool_strides = [4]*num_conv
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
                    structs.extend(struct_.astype(str))
                    losses.append(loss)
                    preds.extend(pred)
                    labels.extend(label_)

                    t.set_description(progress_format.format(epoch_loss))
                    t.update(1)
                except (tf.errors.OutOfRangeError, StopIteration):
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
    if not args.test_only:
        prev_val_loss, best_val_loss = float("inf"), float("inf")

        if (args.max_targets_train == None) and (args.max_decoys_train == None):
            train_num_structs = args.train_sharded.get_num_structures(['ensemble', 'subunit'])
        elif (args.max_targets_train == None):
            train_num_structs = args.train_sharded.get_num_keyed() * args.max_decoys_train
        elif (args.max_decoys_train == None):
            assert False
        else:
            train_num_structs = args.max_targets_train * args.max_decoys_train


        if (args.max_targets_val == None) and (args.max_decoys_val == None):
            val_num_structs = args.val_sharded.get_num_structures(['ensemble', 'subunit'])
        elif (args.max_targets_val == None):
            val_num_structs = args.val_sharded.get_num_keyed() * args.max_decoys_val
        elif (args.max_decoys_val == None):
            assert False
        else:
            val_num_structs = args.max_targets_val * args.max_decoys_val

        train_num_structs *= args.repeat_gen
        #val_num_structs *= args.repeat_gen

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
                feature_psp.dataset_generator,
                args.train_sharded, args.grid_config,
                score_type=args.score_type,
                shuffle=args.shuffle,
                repeat=args.repeat_gen,
                max_targets=args.max_targets_train,
                max_decoys=args.max_decoys_train,
                max_dist_threshold=300.0,
                random_seed=random_seed)

            logging.debug('Creating val generator...')
            val_generator_callable = functools.partial(
                feature_psp.dataset_generator,
                args.val_sharded, args.grid_config,
                score_type=args.score_type,
                shuffle=args.shuffle,
                repeat=1,#*args.repeat_gen,
                max_targets=args.max_targets_val,
                max_decoys=args.max_decoys_val,
                max_dist_threshold=300.0,
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

            ## Save last train and val results
            train_df = pd.DataFrame(
                np.array([train_structs, train_labels, train_preds]).T,
                columns=['structure', 'true', 'pred'],
                )
            train_df['target'] = train_df.structure.apply(
                lambda x: psp_util.get_target_name(x))
            train_df.to_pickle(os.path.join(args.output_dir, 'train_result-{:}.pkl'.format(epoch)))
            __stats('Train Epoch {:}'.format(epoch), train_df)

            val_df = pd.DataFrame(
                np.array([val_structs, val_labels, val_preds]).T,
                columns=['structure', 'true', 'pred'],
                )
            val_df['target'] = val_df.structure.apply(
                lambda x: psp_util.get_target_name(x))
            val_df.to_pickle(os.path.join(args.output_dir, 'val_result-{:}.pkl'.format(epoch)))
            __stats('Val Epoch {:}'.format(epoch), val_df)

            if args.early_stopping and curr_val_loss >= prev_val_loss:
                logging.info("Validation loss stopped decreasing, stopping...")
                break
            else:
                prev_val_loss = curr_val_loss

        logging.info("Finished training")

    ##### Testing
    logging.debug("Run testing")
    if not args.test_only:
        to_use = run_info['best_ckpt'] if args.use_best else ckpt
    else:
        with open(os.path.join(args.model_dir, 'run_info.json')) as f:
            run_info = json.load(f)
        to_use = run_info['best_ckpt']
        saver = tf.train.import_meta_graph(to_use + '.meta')

    logging.info("Using {:} for testing".format(to_use))
    saver.restore(sess, to_use)

    test_generator_callable = functools.partial(
        feature_psp.dataset_generator,
        args.test_sharded, args.grid_config,
        score_type=args.score_type,
        shuffle=args.shuffle,
        repeat=1,
        max_targets=args.max_targets_test,
        max_decoys=args.max_decoys_test,
        max_dist_threshold=None,
        random_seed=args.random_seed)

    if (args.max_targets_test == None) and (args.max_decoys_test == None):
        test_num_structs = args.test_sharded.get_num_structures(['ensemble', 'subunit'])
    elif (args.max_targets_test == None):
        test_num_structs = args.test_sharded.get_num_keyed() * args.max_decoys_test
    elif (args.max_decoys_test == None):
        assert False
    else:
        test_num_structs = args.max_targets_test * args.max_decoys_test

    logging.info("Start testing with {:} structs".format(test_num_structs))


    test_structs, test_preds, test_labels, _, test_loss = __loop(
        test_generator_callable, 'test', num_iters=test_num_structs)
    logging.info("Finished testing")

    test_df = pd.DataFrame(
        np.array([test_structs, test_labels, test_preds]).T,
        columns=['structure', 'true', 'pred'],
        )

    test_df.to_pickle(os.path.join(args.output_dir, 'test_result.pkl'))
    test_df['target'] = test_df.structure.apply(
        lambda x: psp_util.get_target_name(x))
    test_df.to_pickle(os.path.join(args.output_dir, 'test_result.pkl'))
    __stats('Test', test_df)


def create_train_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_sharded', type=str,
        default=os.environ['PSP_TRAIN_SHARDED'])
    parser.add_argument(
        '--val_sharded', type=str,
        default=os.environ['PSP_VAL_SHARDED'])
    parser.add_argument(
        '--test_sharded', type=str,
        default=os.environ['PSP_TEST_SHARDED'])
    parser.add_argument(
        '--output_dir', type=str,
        default=os.environ['MODEL_DIR'])

    # Training parameters
    parser.add_argument('--max_targets_train', type=int, default=None)
    parser.add_argument('--max_targets_val', type=int, default=None)
    parser.add_argument('--max_targets_test', type=int, default=None)

    parser.add_argument('--max_decoys_train', type=int, default=None)
    parser.add_argument('--max_decoys_val', type=int, default=None)
    parser.add_argument('--max_decoys_test', type=int, default=None)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)
    parser.add_argument('--top_nn_drop_rate', type=float, default=0.5)
    parser.add_argument('--top_nn_activation', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--repeat_gen', type=int, default=1)

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

    # Model parameters
    parser.add_argument('--score_type', type=str, default='gdt_ts')

    # Test only
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--model_dir', type=str, default=None)

    return parser


def main():
    parser = create_train_parser()
    args = parser.parse_args()

    args.__dict__['grid_config'] = feature_psp.grid_config

    if args.test_only:
        with open(os.path.join(args.model_dir, 'config.json')) as f:
            model_config = json.load(f)
            args.num_conv = model_config['num_conv']
            args.use_batch_norm = model_config['use_batch_norm']
            if 'grid_config' in model_config:
                args.__dict__['grid_config'] = util.dotdict(
                    model_config['grid_config'])

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.info("Running 3D CNN PSP training...")

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

    logging.info("\n" + str(json.dumps(args.__dict__, indent=4)) + "\n")

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    args.train_sharded = sh.Sharded.load(args.train_sharded)
    args.val_sharded = sh.Sharded.load(args.val_sharded)
    args.test_sharded = sh.Sharded.load(args.test_sharded)

    logging.info("Writing all output to {:}".format(args.output_dir))
    with tf.Session() as sess:
        np.random.seed(args.random_seed)
        tf.set_random_seed(args.random_seed)
        train_model(sess, args)


if __name__ == '__main__':
    main()
