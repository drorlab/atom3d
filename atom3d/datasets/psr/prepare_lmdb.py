import collections as col
import logging
import os
import random
import re
import sys

import click
import torch

import atom3d.datasets.datasets as da
import atom3d.datasets.psr.util as util
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
            return self._scores.loc[key]
        return None

    def __call__(self, x, error_if_missing=False):
        x['scores'] = self._lookup(x['file_path'])
        if x['scores'] is None and error_if_missing:
            raise RuntimeError(f'Unable to find scores for {x["file_path"]}')
        return x


@click.command(help='Prepare psr dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('train_txt', type=click.Path())
@click.argument('val_txt', type=click.Path())
@click.argument('test_txt', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--score_path', type=click.Path(exists=True), default=None)
@click.option('--structures_per_protein', type=int, default=50)
def prepare(input_file_path, train_txt, val_txt, test_txt,
            output_root, score_path, structures_per_protein):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    # Assume PDB filetype.
    filetype = 'pdb'

    scores = Scores(score_path) if score_path else None

    logger.info(f'Splitting indices')
    file_list = fi.find_files(input_file_path, fo.patterns[filetype])
    random.shuffle(file_list)
    target_indices = col.defaultdict(list)
    for i, f in enumerate(file_list):
        target = util.get_target_name(f)
        if len(target_indices[target]) >= structures_per_protein:
            continue
        target_indices[target].append(i)

    dataset = da.load_dataset(file_list, filetype, transform=scores)

    with open(train_txt, 'r') as f:
        train_list = [x.strip() for x in f.readlines()]
    with open(val_txt, 'r') as f:
        val_list = [x.strip() for x in f.readlines()]
    with open(test_txt, 'r') as f:
        test_list = [x.strip() for x in f.readlines()]

    logger.info(f'Writing train')
    train_indices = [f for target in train_list for f in target_indices[target]]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    da.make_lmdb_dataset(train_dataset, os.path.join(output_root, 'train'))

    logger.info(f'Writing val')
    val_indices = [f for target in val_list for f in target_indices[target]]
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    da.make_lmdb_dataset(val_dataset, os.path.join(output_root, 'val'))

    logger.info(f'Writing test')
    test_indices = [f for target in test_list for f in target_indices[target]]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    da.make_lmdb_dataset(test_dataset, os.path.join(output_root, 'test'))


if __name__ == "__main__":
    prepare()
