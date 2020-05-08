import argparse
import subprocess
import os

import dotenv as de
de.load_dotenv(de.find_dotenv())

import numpy as np
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import examples.cnn3d.load_data as load_data
import examples.cnn3d.model as model
import examples.cnn3d.feature_psp as feature_psp


def predict(sess, maps_placeholder, is_training, logits, filename,
            print_per_res_score=False):
    print('# Scoring ' + filename)
    # mapping protein
    pred_dataset = feature_psp.pdb_to_subgrid_dataset(filename)
    # compute prediction
    feed_dict = {maps_placeholder: pred_dataset.maps, is_training: False}
    preds = sess.run(logits, feed_dict=feed_dict)
    if print_per_res_score:
        for i in range(pred_dataset.num_res):
            outline='RES {:4d} {:c} {:5.4f}'.format(
                pred_dataset.meta[i][0], pred_dataset.meta[i][1], preds[i])
            print(outline)
    # global score
    global_score = np.mean(preds)
    print('Global score: {:5.4f}'.format(global_score))
    return global_score


def main(args):
    sess = tf.Session()
    print('Restore existing model: %s' % os.environ['MODEL_PATH'])
    saver = tf.train.import_meta_graph(os.environ['MODEL_PATH'] + '.meta')
    saver.restore(sess, os.environ['MODEL_PATH'])

    graph = tf.get_default_graph()

    # getting placeholder for input data and output
    maps_placeholder = graph.get_tensor_by_name('main_input:0')
    is_training = graph.get_tensor_by_name('is_training:0')
    logits = graph.get_tensor_by_name("main_output:0")

    if args.structure != None :
        predict(sess, maps_placeholder, is_training, logits,
                args.structure, args.print_res)
    if args.directory != None :
        for filename in os.listdir(args.directory):
            predict(sess, maps_placeholder, is_training, logits,
                    args.directory+'/'+filename,
                    args.print_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--directory',
        type=str,
        help='Path to the validation data'
    )
    parser.add_argument(
        '-s',
        '--structure',
        type=str,
        help='Path to the structure to score (in pdb format)'
    )
    parser.add_argument(
        '-r',
        '--print_res',
        action='store_true',
        default=False,
        help='If specified, print per-residue score')
    args = parser.parse_args()
    main(args)
