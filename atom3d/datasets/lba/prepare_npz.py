import collections as col
import logging
import os
import re
import sys

import click
import numpy as np
import pandas as pd 

import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo


logger = logging.getLogger(__name__)


def _filter(dataset, maxnumat):
    # By default, keep all frames
    indices = []
    # Find indices of molecules to keep
    for i,item in enumerate(dataset):
        # Concatenate all relevant atoms
        atom_frames = [item['atoms_pocket'],item['atoms_ligand']]
        atoms = pd.concat(atom_frames, ignore_index=True)
        # Count heavy atoms
        num_heavy_atoms = sum(atoms['element'] != 'H')
        # Does the structure contain undesired elements
        allowed = ['H','C','N','O','S','Zn','Cl','F','P','Mg']
        unwanted_elements = set(atoms['element']) - set(allowed)
        # add the index
        if num_heavy_atoms <= maxnumat and len(unwanted_elements) == 0:
            indices.append(i)
    indices = np.array(indices, dtype=int)
    return indices


def _write_npz(dataset, filename, indices, drop):
    # Get the coordinates
    save_dict = da.extract_coordinates_as_numpy_arrays(dataset, indices, 
        atom_frames=['atoms_pocket','atoms_ligand'], drop_elements=['H'])
    # Add the label data 
    save_dict['neglog_aff'] = np.array([dataset[i]['scores']['data'][0] for i in indices])
    # Save the data
    np.savez_compressed(filename,**save_dict)
    

class UpdateTypes():
    def __init__(self, df_keys):
        self.df_keys = df_keys
    def __call__(self,x):
        for key in self.df_keys:
            if key in x and type(x[key]) != pd.DataFrame: 
                x[key] = pd.DataFrame(**x[key])
        return x


@click.command(help='Prepare SMP dataset')
@click.argument('input_root', type=click.Path())
@click.argument('output_file_path', type=click.Path())
@click.argument('maxnumat', default=500, type=int)
@click.option('--split', '-s', is_flag=True)
@click.option('--droph', '-d', is_flag=True)
def prepare(input_root, output_file_path, split, droph, maxnumat):
    # Function for input and output path
    out_path = lambda f: os.path.join(output_file_path,f)
    inp_path = lambda f: os.path.join(input_root,f)
    # Define which elements to drop
    drop = []
    if droph: drop.append('H')
    # Logger
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format='%(asctime)s %(levelname)s %(process)d: '+'%(message)s')
    if split: # use the split datasets 
        logger.info(f'Processing datasets from {input_root:}.')
        # Training set
        logger.info(f'Processing training dataset...')
        dataset = da.LMDBDataset(inp_path('train'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
        indices = _filter(dataset, maxnumat)
        _write_npz(dataset, out_path('train.npz'), indices, drop)
        # Validation set
        logger.info(f'Processing validation dataset...')
        dataset = da.LMDBDataset(inp_path('val'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
        indices = _filter(dataset, maxnumat)
        _write_npz(dataset, out_path('valid.npz'), indices, drop)
        # Test set
        logger.info(f'Processing test dataset...')
        dataset = da.LMDBDataset(inp_path('test'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
        indices = _filter(dataset, maxnumat)
        _write_npz(dataset, out_path('test.npz'), indices, drop)
    else: # use the full data set
        logger.info(f'Processing full dataset from {input_root:}...')
        dataset = da.LMDBDataset(inp_path('all'), transform=UpdateTypes(['atoms_pocket','atoms_ligand']))
        indices = _filter(dataset, maxnumat)
        _write_npz(dataset, out_path('all.npz'), indices, drop)


if __name__ == "__main__":
    prepare()

