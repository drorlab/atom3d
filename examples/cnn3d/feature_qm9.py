import os
import tqdm

import numpy as np
import pandas as pd

import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util


grid_config = util.dotdict({
    # Mapping from elements to position in channel dimension.
    'element_mapping': {
        'H': 0,
        'C': 1,
        'O': 2,
        'N': 3,
        'F': 4,
    },
    # Radius of the grids to generate, in angstroms.
    'radius': 7.5,
    # Resolution of each voxel, in angstroms.
    'resolution': 1.0,
    # Number of directions to apply for data augmentation.
    'num_directions': 20,
    # Number of rolls to apply for data augmentation.
    'num_rolls': 20,
})


def read_labels(labels_filename, mol_ids, label_type):
    """
    Return a pandas DataFrame containing labels for all pdbs with header
    <pdb> and <label>.
    """
    labels_df = pd.read_csv(labels_filename, delimiter=',', engine='python').dropna()
    return labels_df[labels_df.mol_id.isin(mol_ids)][['mol_id', label_type]].reset_index(drop=True)


def df_to_feature(mol_df, grid_config, random_seed=None):
    pos = mol_df[['x', 'y', 'z']].astype(np.float32)
    center = util.get_center(pos)

    rot_mat = subgrid_gen.gen_rot_matrix(grid_config, random_seed=random_seed)
    grid = subgrid_gen.get_grid(
        mol_df, center, config=grid_config, rot_mat=rot_mat)
    return grid


def dataset_generator(data_filename, labels_filename, grid_config, label_type,
                      shuffle=True, repeat=None, max_mols=None, random_seed=None):

    data_df = pd.read_hdf(data_filename, 'structures')
    all_mol_ids = data_df.structure.unique()

    labels_df = read_labels(labels_filename, all_mol_ids, label_type)

    # Some mol_ids might not have labels, so we need to prune them
    all_mol_ids = labels_df.mol_id.unique()
    data_df = data_df[data_df.structure.isin(all_mol_ids)]

    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        if shuffle:
            p = np.random.permutation(len(all_mol_ids))
            all_mol_ids = all_mol_ids[p]
        mol_ids = all_mol_ids if max_mols is None else all_mol_ids[:max_mols]

        for i, mol_id in enumerate(mol_ids):
            mol_df = data_df[data_df.structure == mol_id]

            feature = df_to_feature(mol_df, grid_config, random_seed)
            label = labels_df[labels_df.mol_id == mol_id][label_type].values

            yield mol_id, feature, label


def get_data_stats(data_filename):
    """
    Get the furthest distance from the molecule's center and the number of
    atoms for each molecule in the dataset.
    """
    data_df = pd.read_hdf(data_filename, 'structures')

    data = []
    for mol_id, mol_df in data_df.groupby(['structure']):
        pos = mol_df[['x', 'y', 'z']].astype(np.float32)
        max_dist = util.get_max_distance_from_center(pos, util.get_center(pos))
        num_atoms = mol_df.shape[0]
        data.append((mol_id, max_dist, num_atoms))

    df = pd.DataFrame(data, columns=['mol_id', 'max_dist', 'num_atoms'])
    df = df.sort_values(by=['max_dist', 'num_atoms'],
                        ascending=[False, False]).reset_index(drop=True)
    print(df.describe())

    print(df[df.max_dist < 7.5].shape[0]*100.0/df.shape[0])
    return df


if __name__ == "__main__":
    base_filename = '/oak/stanford/groups/rondror/users/mvoegele/atom3d/data/qm9/hdf5/test'
    data_filename = base_filename + '.h5'
    labels_filename = base_filename + '.csv'

    data_stats_df = get_data_stats(data_filename)

    print('Testing qm9 feature generator')
    gen = dataset_generator(
        data_filename, labels_filename, grid_config, label_type='alpha',
        shuffle=True, repeat=1, max_mols=10)

    for i, (mol_id, feature, label) in enumerate(gen):
        print('Generating feature {:} {:} -> feature {:}, label {:}'.format(
            i, mol_id, feature.shape, label.shape))
