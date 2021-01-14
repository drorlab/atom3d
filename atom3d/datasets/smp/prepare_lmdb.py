import collections as col
import logging
import os
import re
import sys

import click
import numpy as np

import atom3d.datasets.datasets as da
import atom3d.datasets.psr.util as util
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo


logger = logging.getLogger(__name__)


def _add_data_with_subtracted_thermochem_energy(x):
    """
    Adds energies with subtracted thermochemical energies to the data list.
    We only need this for the QM9 dataset (SMP).
    """
    data = x['labels']
    # per-atom thermochem. energies for U0 [Ha], U [Ha], H [Ha], G [Ha], Cv [cal/(mol*K)]
    # https://figshare.com/articles/dataset/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643
    thchem_en = {'H': [-0.500273, -0.498857, -0.497912, -0.510927, 2.981],
                 'C': [-37.846772, -37.845355, -37.844411, -37.861317, 2.981],
                 'N': [-54.583861, -54.582445, -54.581501, -54.598897, 2.981],
                 'O': [-75.064579, -75.063163, -75.062219, -75.079532, 2.981],
                 'F': [-99.718730, -99.717314, -99.716370, -99.733544, 2.981]}
    # Count occurence of each element in the molecule
    counts = x['atoms']['element'].value_counts()
    # Calculate and subtract thermochemical energies
    u0_atom = data[10] - np.sum([c * thchem_en[el][0] for el, c in counts.items()])  # U0
    u_atom  = data[11] - np.sum([c * thchem_en[el][1] for el, c in counts.items()])  # U
    h_atom  = data[12] - np.sum([c * thchem_en[el][2] for el, c in counts.items()])  # H
    g_atom  = data[13] - np.sum([c * thchem_en[el][3] for el, c in counts.items()])  # G
    cv_atom = data[14] - np.sum([c * thchem_en[el][4] for el, c in counts.items()])  # Cv
    # Append new data
    data += [u0_atom, u_atom, h_atom, g_atom, cv_atom]
    return x


def _write_split_indices(split_txt, lmdb_ds, output_txt):
    with open(split_txt, 'r') as f:
        split_set = set([x.strip() for x in f.readlines()])
    # Check if the target in id is in the desired target split set
    split_ids = list(filter(lambda idi: idi in split_set, lmdb_ds.ids()))
    # Convert ids into lmdb numerical indices and write into txt file
    split_indices = lmdb_ds.ids_to_indices(split_ids)
    with open(output_txt, 'w') as f:
        f.write(str('\n'.join([str(i) for i in split_indices])))
    return split_indices


@click.command(help='Prepare SMP dataset')
@click.argument('input_file_path', type=click.Path())
@click.argument('output_root', type=click.Path())
@click.option('--split', '-s', is_flag=True)
@click.option('--train_txt', '-tr', type=click.Path(exists=True), default=None)
@click.option('--val_txt', '-v', type=click.Path(exists=True), default=None)
@click.option('--test_txt', '-t', type=click.Path(exists=True), default=None)
def prepare(input_file_path, output_root, split, train_txt, val_txt, test_txt):
    # Logger
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    # Assume GDB-specific version of XYZ format.
    filetype = 'xyz-gdb'
    # Compile a list of the input files
    file_list = fi.find_files(input_file_path, fo.patterns[filetype])
    # Write the LMDB dataset
    lmdb_path = os.path.join(output_root, 'all')
    logger.info(f'Creating lmdb dataset into {lmdb_path:}...')
    dataset = da.load_dataset(file_list, filetype, transform=_add_data_with_subtracted_thermochem_energy)
    da.make_lmdb_dataset(dataset, lmdb_path)
    # Only continue if we want to write split datasets
    if not split:
        return
    logger.info(f'Splitting indices...\n')
    # Load the dataset that has just been created
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')
    # Determine and write out the split indices
    indices_train = _write_split_indices(train_txt, lmdb_ds, os.path.join(output_root, 'train_indices.txt'))
    indices_val = _write_split_indices(val_txt, lmdb_ds, os.path.join(output_root, 'val_indices.txt'))
    indices_test = _write_split_indices(test_txt, lmdb_ds, os.path.join(output_root, 'test_indices.txt'))
    # Write the split datasets
    train_dataset, val_dataset, test_dataset = spl.split(lmdb_ds, indices_train, indices_val, indices_test)
    da.make_lmdb_dataset(train_dataset, os.path.join(output_root, 'train'))
    da.make_lmdb_dataset(val_dataset, os.path.join(output_root, 'val'))
    da.make_lmdb_dataset(test_dataset, os.path.join(output_root, 'test'))


if __name__ == "__main__":
    prepare()

