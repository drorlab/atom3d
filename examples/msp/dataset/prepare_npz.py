import collections as col
import logging
import os
import re
import sys

import click
import numpy as np
import scipy as sp
import scipy.spatial
import pandas as pd 

import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo


logger = logging.getLogger(__name__)


class EnvironmentSelection(object):
    """
    Selects a region of protein coordinates within a certain distance from the alpha carbon of the mutated residue.

    :param df: Atoms data
    :type df: pandas.DataFrame 
    :param dist: Distance from the alpha carbon of the mutated residue
    :type dist: float

    :return new_df: Transformed atoms data
    :rtype new_df: pandas.DataFrame

    """
    def __init__(self, dist):
        self._dist = dist

    def _get_mutation(self, x):
        mutation = x['id'].split('_')[-1]
        chain = mutation[1]
        resid = int(mutation[2:-1])
        original_resname = mutation[0]
        mutation_resname = mutation[-1]
        return chain, resid

    def _select_env(self, df, chain, resid):
        # Find the C-alpha atom of the mutated residue
        mutated = df[(df.chain == chain) & (df.residue == resid)]
        mut_c_a = mutated[mutated.name == 'CA']
        # Define the protein atoms
        protein = df
        # extract coordinates
        muta_coords = np.array([mut_c_a.x, mut_c_a.y, mut_c_a.z]).T
        prot_coords = np.array([protein.x, protein.y, protein.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(prot_coords)
        key_pts = kd_tree.query_ball_point(muta_coords, r=self._dist, p=2.0)
        key_pts = np.unique([k for l in key_pts for k in l])
        # Construct the new data frame
        new_df = pd.concat([ protein.iloc[key_pts] ], ignore_index=True)
        return new_df

    def __call__(self, x):
        # Extract mutation info from the ID
        chain, resid = self._get_mutation(x)
        # Select environment in original data frame
        x['original_atoms'] = self._select_env(x['original_atoms'], chain, resid)
        # Select environment in mutated data frame
        x['mutated_atoms'] = self._select_env(x['mutated_atoms'], chain, resid)
        return x 


def _write_npz(dataset, filename, drop):
    keys = ['index', 'num_atoms', 'charges', 'positions']
    # Load original atoms
    ori = da.extract_coordinates_as_numpy_arrays(dataset,  
          atom_frames=['original_atoms'], drop_elements=drop)
    for k in keys: ori['original_'+k] = ori.pop(k)
    # Load mutated atoms
    mut = da.extract_coordinates_as_numpy_arrays(dataset, 
          atom_frames=['mutated_atoms'], drop_elements=drop)
    for k in keys: mut['mutated_'+k] = mut.pop(k)
    # Merge datasets with atoms
    save_dict = {**ori, **mut}
    # Add labels
    labels = [item['label'] for item in dataset]
    save_dict['label'] = np.array(labels, dtype=int)
    # Save the data
    np.savez_compressed(filename,**save_dict)
    return save_dict 


@click.command(help='Prepare MSP dataset')
@click.argument('input_root', type=click.Path())
@click.argument('output_file_path', type=click.Path())
@click.argument('radius', default=6, type=float)
@click.option('--split', '-s', is_flag=True)
@click.option('--droph', '-d', is_flag=True)
def prepare(input_root, output_file_path, split, droph, radius):
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
        dataset = da.LMDBDataset(inp_path('train'), transform=EnvironmentSelection(radius))
        _save_dict = _write_npz(dataset, out_path('train.npz'), drop)
        # Validation set
        logger.info(f'Processing validation dataset...')
        dataset = da.LMDBDataset(inp_path('val'), transform=EnvironmentSelection(radius))
        _save_dict =_write_npz(dataset, out_path('valid.npz'), drop)
        # Test set
        logger.info(f'Processing test dataset...')
        dataset = da.LMDBDataset(inp_path('test'), transform=EnvironmentSelection(radius))
        _save_dict =_write_npz(dataset, out_path('test.npz'), drop)
    else: # use the full data set
        logger.info(f'Processing full dataset from {input_root:}...')
        dataset = da.LMDBDataset(inp_path('all'), transform=EnvironmentSelection(radius))
        _save_dict =_write_npz(dataset, out_path('all.npz'), drop)


if __name__ == "__main__":
    prepare()

