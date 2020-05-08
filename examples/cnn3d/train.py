from __future__ import division, print_function, absolute_import

import argparse
import json
import logging
import os
import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

import atom3d.util.shard as sh
import atom3d.psp.util as psp_util
import examples.cnn3d.model as model
import examples.cnn3d.feature_psp as feature_psp


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

    # Save metrics.
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


# Construct model and loss
def conv_model(feature, target, is_training):
    per_res_scores, _, _, _ = model.scoring_model(
        model.NUM_RETYPE,
        feature,
        is_training=is_training,
        batch_norm=True,
        validation='softplus',
        final_activation='sigmoid')
    # Predict global score (i.e. average across local predictions for each
    # residue within a structure)
    predict = tf.reduce_mean(per_res_scores, keepdims=True, name='predict')
    # Loss
    loss = tf.losses.mean_squared_error(target, predict)
    return predict, loss


def train_model(sess, args):
    # tf Graph input
    # Subgrid maps for each residue in a protein
    logging.debug('Create input placeholder...')
    feature_placeholder = tf.placeholder(
        tf.float32,
        [None, model.GRID_SIZE, model.GRID_SIZE, model.GRID_SIZE, model.NB_TYPE],
        name='main_input')
    label_placeholder = tf.placeholder(tf.float32, [1], 'label')
    training_placeholder = tf.placeholder(tf.bool, shape=[], name='is_training')

    # Define loss and optimizer
    logging.debug('Define loss and optimizer...')
    predict_op, loss_op = conv_model(
        feature_placeholder, label_placeholder, training_placeholder)
    logging.debug('Generate training ops...')
    train_op = model.training(loss_op, args.learning_rate)

    # Initialize the variables (i.e. assign their default value)
    logging.debug('Initialize global variables...')
    init = tf.global_variables_initializer()

    # Create saver and summaries.
    logging.debug('Initialize global variables...')
    saver = tf.train.Saver(max_to_keep=100000)

    def __loop(generator, mode, num_iters):
        structs, losses, preds, labels = [], [], [], []
        epoch_loss = 0
        progress_format = mode + ' loss: {:6.6f}'

        # Loop over all batches (one batch is all feature for 1 protein)
        with tqdm.tqdm(total=num_iters, desc=progress_format.format(0)) as t:
            for i in range(num_iters):
                struct_, feature_, label_ = next(generator)
                _, pred, loss = sess.run(
                    [train_op, predict_op, loss_op],
                    feed_dict={feature_placeholder: feature_,
                               label_placeholder: label_,
                               training_placeholder: (mode == 'train')})
                epoch_loss += (np.mean(loss) - epoch_loss) / (i + 1)

                structs.append(struct_)
                losses.append(loss)
                preds.append(pred)
                labels.append(label_)

                t.set_description(progress_format.format(epoch_loss))
                t.update(1)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        return structs, preds, labels, losses, epoch_loss

    # Run the initializer
    sess.run(init)

    ##### Training + validation
    prev_val_loss, best_val_loss = float("inf"), float("inf")

    train_generator = feature_psp.subgrid_dataset_generator(
        args.train_sharded, args.scores_dir, args.score_type, args.nb_type, args.
        grid_size, args.shuffle, args.random_seed, repeat=args.num_epochs,
        num_iters=args.num_iters_train, max_res=args.max_res)

    val_generator = feature_psp.subgrid_dataset_generator(
        args.val_sharded, args.scores_dir, args.score_type, args.nb_type,
        args.grid_size, args.shuffle, args.random_seed,
        repeat=args.num_epochs, num_iters=args.num_iters_val,
        max_res=args.max_res)

    train_total_structs = sh.get_num_structures(args.train_sharded)
    val_total_structs = sh.get_num_structures(args.val_sharded)

    if args.num_iters_train == None:
        train_num_structs = train_total_structs
    else:
        train_num_structs = args.num_iters_train
    if args.num_iters_val == None:
        val_num_structs = val_total_structs
    else:
        val_num_structs = args.num_iters_val

    logging.info("Start training with {:} structs for train and {:} structs for val per epoch".format(
        train_num_structs, val_num_structs))
    logging.info("Total train: {:} structs , total val: {:} structs".format(
        train_total_structs, val_total_structs))

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

    val_losses = []
    for epoch in range(1, args.num_epochs+1):
        logging.info('Epoch {:}'.format(epoch))
        # Training
        _, _, _, _, curr_train_loss = __loop(
            train_generator, 'train', num_iters=train_num_structs)
        # Validation
        _, _, _, _, curr_val_loss = __loop(
            val_generator, 'val', num_iters=val_num_structs)

        val_losses.append(curr_val_loss)
        __update_and_write_run_info('val_losses', val_losses)

        if args.use_best or args.early_stopping:
            prev_val_loss = curr_val_loss
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


    logging.info("Finished training")

    ##### Testing
    to_use = run_info['best_ckpt'] if args.use_best else ckpt
    logging.info("Using {:} for testing".format(to_use))
    saver.restore(sess, to_use)

    test_generator = feature_psp.subgrid_dataset_generator(
        args.test_sharded, args.scores_dir, args.score_type, args.nb_type,
        args.grid_size, shuffle=False, random_seed=None, repeat=None,
        num_iters=args.num_iters_test, max_res=args.max_res)

    test_total_structs = sh.get_num_structures(args.test_sharded)
    if args.num_iters_test == None:
        test_num_structs = test_total_structs
    else:
        test_num_structs = args.num_iters_test

    logging.info("Start testing with {:} structs out of {:} structs".format(
        test_num_structs, test_total_structs))

    test_structs, test_preds, test_labels, test_losses, test_loss = __loop(
        test_generator, 'test', num_iters=test_num_structs)
    logging.info("Finished testing")

    test_df = pd.DataFrame(
        np.array([test_structs, test_labels, test_preds, test_losses]).T,
        columns=['structure', 'true', 'pred', 'loss'],
        )

    test_df['target'] = test_df.structure.apply(
        lambda x: psp_util.get_target_name(x))
    test_df.to_pickle(os.path.join(args.output_dir, 'test_result.pkl'))

    # Compute global correlations
    res = compute_global_correlations(test_df)
    logging.info(
        '\nCorrelations (Pearson, Kendall, Spearman)\n'
        '    per-target averaged median: ({:.3f}, {:.3f}, {:.3f})\n'
        '    per-target averaged mean: ({:.3f}, {:.3f}, {:.3f})\n'
        '    all averaged: ({:.3f}, {:.3f}, {:.3f})'.format(
        float(res["per_target_median_pearson"]),
        float(res["per_target_median_kendall"]),
        float(res["per_target_median_spearman"]),
        float(res["per_target_mean_pearson"]),
        float(res["per_target_mean_kendall"]),
        float(res["per_target_mean_spearman"]),
        float(res["all_pearson"]),
        float(res["all_kendall"]),
        float(res["all_spearman"])))


def create_train_parser():
    parser = argparse.ArgumentParser(description='Abinitio folding')

    parser.add_argument(
        '--scores_dir', type=str,
        default='/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/labels/scores')
    parser.add_argument(
        '--train_sharded', type=str,
        default='/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_20/train_decoy_20@100')
    parser.add_argument(
        '--val_sharded', type=str,
        default='/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_20/val_decoy_20@10')
    parser.add_argument(
        '--test_sharded', type=str,
        default='/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_20/test_decoy_20@20')
    parser.add_argument(
        '--output_dir', type=str,
        default='/scratch/users/psuriana/atom3d/model')

    # Training parameters
    parser.add_argument('--max_res', type=int, default=300)
    parser.add_argument('--num_iters_train', type=int, default=100)
    parser.add_argument('--num_iters_val', type=int, default=100)
    parser.add_argument('--num_iters_test', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--use_best', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=1333)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--unobserved', action='store_true', default=False)

    # Model parameters
    parser.add_argument('--score_type', type=str, default='gdt_ts')
    parser.add_argument('--nb_type', type=int, default=169)
    parser.add_argument('--grid_size', type=int, default=24)

    return parser


def main():
    parser = create_train_parser()
    args = parser.parse_args()

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

    logging.info("\n" + str(args) + "\n")

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    logging.info("Writing all output to {:}".format(args.output_dir))
    with tf.Session() as sess:
        train_model(sess, args)


if __name__ == '__main__':
    main()
