import collections as col
import numpy as np
import pandas as pd
import logging
import os
import sys

import click

import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo
import util

logger = logging.getLogger(__name__)


class Scores(object):
    """
    Track and lookup PSR score files.
    """
    def __init__(self, data_path):
        self._scores = util.read_labels(data_path, ext='dat')

    def _lookup(self, file_path):
        target = util.get_target_name(file_path)
        decoy = util.get_decoy_name(file_path)
        key = (target, decoy)
        if key in self._scores.index:
            score = self._scores.loc[key].astype(np.float64).squeeze().to_dict()
            score = {key.replace('-','_'): val for key,val in score.items()}
            return key, score
        return key, None

    def __call__(self, x, error_if_missing=False):
        key, x['scores'] = self._lookup(x['file_path'])
        x['id'] = str(key)
        if x['scores'] is None and error_if_missing:
            raise RuntimeError(f'Unable to find scores for {x["file_path"]}')
        return x


def split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, split_dir):
    logger.info(f'Splitting indices, load data from {lmdb_path:}...')
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        # Read list of desired <target, decoy>
        split_set = set(map(tuple, pd.read_csv(split_txt, header=None, dtype=str).values))

        # Check if the <target, decoy> id is in the desired split set
        split_ids = list(filter(lambda id: eval(id) in split_set, lmdb_ds.ids()))
        # Convert ids into lmdb numerical indices and write into txt file
        split_indices = lmdb_ds.ids_to_indices(split_ids)
        with open(output_txt, 'w') as f:
            f.write(str('\n'.join([str(i) for i in split_indices])))
        return split_indices

    logger.info(f'Write results to {split_dir:}...')
    os.makedirs(os.path.join(split_dir, 'indices'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'data'), exist_ok=True)

    indices_train = _write_split_indices(
        train_txt, lmdb_ds, os.path.join(split_dir, 'indices/train_indices.txt'))
    indices_val = _write_split_indices(
        val_txt, lmdb_ds, os.path.join(split_dir, 'indices/val_indices.txt'))
    indices_test = _write_split_indices(
        test_txt, lmdb_ds, os.path.join(split_dir, 'indices/test_indices.txt'))

    train_dataset, val_dataset, test_dataset = spl.split(
        lmdb_ds, indices_train, indices_val, indices_test)
    da.make_lmdb_dataset(train_dataset, os.path.join(split_dir, 'data/train'))
    da.make_lmdb_dataset(val_dataset, os.path.join(split_dir, 'data/val'))
    da.make_lmdb_dataset(test_dataset, os.path.join(split_dir, 'data/test'))


def make_lmdb_dataset(input_file_path, score_path, output_root):
    # Assume PDB filetype.
    filetype = 'pdb'

    scores = Scores(score_path) if score_path else None

    file_list = fi.find_files(input_file_path, fo.patterns[filetype])

    lmdb_path = os.path.join(output_root, 'data')
    os.makedirs(lmdb_path, exist_ok=True)

    logger.info(f'Creating lmdb dataset into {lmdb_path:}...')
    dataset = da.load_dataset(file_list, filetype, transform=scores)
    da.make_lmdb_dataset(dataset, lmdb_path, filter_fn=lambda x: x['scores']['rmsd']==-1.0)
    return lmdb_path


@click.command(help='Prepare psr dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--no_gen', '-ng', is_flag=True,
              help="If specified, we don't process the data to lmdb")
@click.option('--split', '-s', is_flag=True)
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-v', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-t', type=click.Path(exists=True), default=None)
@click.option('--score_path', type=click.Path(exists=True), default=None)
def prepare(input_file_path, output_root, no_gen, split,
            train_txt, val_txt, test_txt, score_path):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    if not no_gen:
        input_file_path = make_lmdb_dataset(input_file_path, score_path, output_root)
        output_root = os.path.join(output_root, 'split')
    if split:
        split_lmdb_dataset(input_file_path, train_txt, val_txt, test_txt, output_root)



if __name__ == "__main__":
    prepare()
