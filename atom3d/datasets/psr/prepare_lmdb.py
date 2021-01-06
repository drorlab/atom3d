import collections as col
import logging
import os
import re
import sys

import click

import atom3d.datasets.datasets as da
import atom3d.datasets.psr.util as util
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo


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
            return key, self._scores.loc[key]
        return None, None

    def __call__(self, x, error_if_missing=False):
        key, x['scores'] = self._lookup(x['file_path'])
        if key is not None:
            x['id'] = str(key)
        if x['scores'] is None and error_if_missing:
            raise RuntimeError(f'Unable to find scores for {x["file_path"]}')
        return x


@click.command(help='Prepare psr dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--split', '-s', is_flag=True)
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-v', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-t', type=click.Path(exists=True), default=None)
@click.option('--score_path', type=click.Path(exists=True), default=None)
def prepare(input_file_path, output_root, split, train_txt, val_txt, test_txt,
            score_path):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    # Assume PDB filetype.
    filetype = 'pdb'

    scores = Scores(score_path) if score_path else None

    file_list = fi.find_files(input_file_path, fo.patterns[filetype])

    lmdb_path = os.path.join(output_root, 'all')
    logger.info(f'Creating lmdb dataset into {lmdb_path:}...')
    dataset = da.load_dataset(file_list, filetype, transform=scores)
    da.make_lmdb_dataset(dataset, lmdb_path)

    if not split:
        return

    logger.info(f'Splitting indices...')
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        with open(split_txt, 'r') as f:
            split_set = set([x.strip() for x in f.readlines()])
        # Check if the target in id is in the desired target split set
        split_ids = list(filter(lambda id: eval(id)[0] in split_set, lmdb_ds.ids()))
        # Convert ids into lmdb numerical indices and write into txt file
        split_indices = lmdb_ds.ids_to_indices(split_ids)
        with open(output_txt, 'w') as f:
            f.write(str('\n'.join([str(i) for i in split_indices])))

    _write_split_indices(train_txt, lmdb_ds, os.path.join(output_root, 'train_indices.txt'))
    _write_split_indices(val_txt, lmdb_ds, os.path.join(output_root, 'val_indices.txt'))
    _write_split_indices(test_txt, lmdb_ds, os.path.join(output_root, 'test_indices.txt'))


if __name__ == "__main__":
    prepare()
