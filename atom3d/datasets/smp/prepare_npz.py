import collections as col
import logging
import os
import re
import sys

import click
import numpy as np

import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo


logger = logging.getLogger(__name__)


label_names = ['A','B','C','mu','alpha','homo','lumo','gap','r2',
               'zpve','u0','u298','h298','g298','cv',
               'u0_atom','u298_atom','h298_atom','g298_atom','cv_atom']

def _write_npz(dataset, filename):
    # Get the coordinates
    save_dict = da.extract_coordinates_as_numpy_arrays(dataset)
    # Add the label data 
    for il,label in enumerate(label_names):
        save_dict[label] = np.array([item['labels'][il] for item in dataset])
    # Save the data
    np.savez_compressed(filename,**save_dict)
    

@click.command(help='Prepare SMP dataset')
@click.argument('input_root', type=click.Path())
@click.argument('output_file_path', type=click.Path())
@click.option('--split', '-s', is_flag=True)
def prepare(input_root, output_file_path, split):
    # Logger
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    if split:
        logger.info(f'Processing datasets from {input_root:}.')
        logger.info(f'Processing training dataset...')
        dataset = da.LMDBDataset(os.path.join(input_root, 'train'))
        _write_npz(dataset, os.path.join(output_file_path,'train.npz'))
        logger.info(f'Processing validation dataset...')
        dataset = da.LMDBDataset(os.path.join(input_root, 'val'))
        _write_npz(dataset, os.path.join(output_file_path,'valid.npz'))
        logger.info(f'Processing test dataset from...')
        dataset = da.LMDBDataset(os.path.join(input_root, 'test'))
        _write_npz(dataset, os.path.join(output_file_path,'test.npz'))
    else:
        logger.info(f'Processing full dataset from {input_root:}...')
        dataset = da.LMDBDataset(os.path.join(input_root, 'all'))
        _write_npz(dataset, os.path.join(output_file_path,'all.npz'))


if __name__ == "__main__":
    prepare()

