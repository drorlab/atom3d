import collections as col
import logging
import os
import random
import re
import sys

import click
import torch

import atom3d.datasets.datasets as da
import atom3d.util.file as fi
import atom3d.util.formats as fo
import atom3d.util.rosetta as ar


logger = logging.getLogger(__name__)

# Canonical splits.
TRAIN = \
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
VAL = ['14b', '14f', '15', '17']
TEST = ['18', '19', '20', '21']


def get_target(f):
    """Extract string target name from filepath."""
    dir_name = os.path.basename(os.path.dirname(f))

    NUMBER_PATTERN = re.compile('_([0-9]{1,2})(_|$|\.)')
    target_number = int(re.search(NUMBER_PATTERN, dir_name).group(1))
    if target_number != 14:
        target = str(target_number)
    else:
        # We keep bound and free denotation if puzzle 14.
        target = str(target_number) + \
            ('b' if 'bound' in dir_name else 'f')
    return target


@click.command(help='Prepare rsr dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--score_path', type=click.Path(exists=True), default=None)
@click.option('--structures_per_rna', type=int, default=1000)
def prepare(input_file_path, output_root, score_path, structures_per_rna):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)

    # Assume PDB filetype.
    filetype = 'pdb'

    scores = ar.Scores(score_path) if score_path else None

    logger.info(f'Splitting indices')
    file_list = fi.find_files(input_file_path, fo.patterns[filetype])
    random.shuffle(file_list)
    target_indices = col.defaultdict(list)
    for i, f in enumerate(file_list):
        target = get_target(f)
        if len(target_indices[target]) >= structures_per_rna:
            continue
        target_indices[target].append(i)

    dataset = da.load_dataset(file_list, filetype, transform=scores)

    logger.info(f'Writing train')
    train_indices = [f for target in TRAIN for f in target_indices[target]]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    da.make_lmdb_dataset(train_dataset, os.path.join(output_root, 'train'))

    logger.info(f'Writing val')
    val_indices = [f for target in VAL for f in target_indices[target]]
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    da.make_lmdb_dataset(val_dataset, os.path.join(output_root, 'val'))

    logger.info(f'Writing test')
    test_indices = [f for target in TEST for f in target_indices[target]]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    da.make_lmdb_dataset(test_dataset, os.path.join(output_root, 'test'))


if __name__ == "__main__":
    prepare()
