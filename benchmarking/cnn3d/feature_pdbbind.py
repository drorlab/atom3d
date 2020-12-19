import os

import dotenv as de
import numpy as np
import pandas as pd

import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util

de.load_dotenv(de.find_dotenv(usecwd=True))


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
    'radius': 20.0,
    # Resolution of each voxel, in angstroms.
    'resolution': 1.0,
    # Number of directions to apply for data augmentation.
    'num_directions': 20,
    # Number of rolls to apply for data augmentation.
    'num_rolls': 20,
})


def read_labels(labels_filename, pdbcodes):
    """
    Return a pandas DataFrame containing labels for all pdbs with header
    <pdb> and <label>.
    """
    labels_df = pd.read_csv(labels_filename, delimiter=',', engine='python').dropna()
    return labels_df[labels_df.pdb.isin(pdbcodes)].reset_index(drop=True)


def read_split(split_filename):
    """
    Return a list of pdb codes included in the split.
    """
    with open(split_filename, 'r') as f:
        pdbcodes = [t.strip() for t in f.readlines()]
    return pdbcodes


def df_to_feature(struct_df, grid_config, random_seed=None):
    pos = struct_df[['x', 'y', 'z']].astype(np.float32)
    # Use center of ligand for subgrid center
    ligand_pos = struct_df[struct_df.chain == 'LIG'][['x', 'y', 'z']].astype(
        np.float32)
    ligand_center = util.get_center(ligand_pos)

    rot_mat = subgrid_gen.gen_rot_matrix(grid_config, random_seed=random_seed)
    grid = subgrid_gen.get_grid(
        struct_df, ligand_center, config=grid_config, rot_mat=rot_mat)
    return grid


def dataset_generator(data_filename, split_filename, labels_filename,
                      grid_config, shuffle=True, repeat=None, max_pdbs=None,
                      random_seed=None):
    all_pdbcodes = read_split(split_filename)
    labels_df = read_labels(labels_filename, all_pdbcodes)

    # Some pdbcodes might not have labels, so we need to prune them
    all_pdbcodes = labels_df.pdb.unique()

    data_df = pd.read_hdf(data_filename, 'structures')
    data_df = data_df[data_df.ensemble.isin(all_pdbcodes)]

    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        if shuffle:
            p = np.random.permutation(len(all_pdbcodes))
            all_pdbcodes = all_pdbcodes[p]
        pdbcodes = all_pdbcodes if max_pdbs is None else all_pdbcodes[:max_pdbs]

        for i, pdbcode in enumerate(pdbcodes):
            struct_df = data_df[data_df.ensemble == pdbcode]

            feature = df_to_feature(struct_df, grid_config, random_seed)
            label = labels_df[labels_df.pdb == pdbcode].label.values

            yield pdbcode, feature, label


def get_data_stats(data_filename):
    """
    Get the furthest distance from the ligand's center and the number of
    atoms for each structure in the dataset.
    """
    data_df = pd.read_hdf(data_filename, 'structures')

    data = []
    for pdbcode, struct_df in data_df.groupby(['structure']):
        pos = struct_df[['x', 'y', 'z']].astype(np.float32)

        ligand_pos = struct_df[struct_df.chain == 'LIG'][['x', 'y', 'z']].astype(
            np.float32)
        ligand_center = util.get_center(ligand_pos)

        max_dist = util.get_max_distance_from_center(pos, ligand_center)
        num_atoms = struct_df.shape[0]
        data.append((pdbcode, max_dist, num_atoms))

    df = pd.DataFrame(data, columns=['pdbcode', 'max_dist', 'num_atoms'])
    df = df.sort_values(by=['max_dist', 'num_atoms'],
                        ascending=[False, False]).reset_index(drop=True)
    print(df.describe())

    print(df[df.max_dist < 20].shape[0]*100.0/df.shape[0])
    return df


if __name__ == "__main__":
    data_filename = os.environ['PDBBIND_DATA_FILENAME']
    labels_filename = os.environ['PDBBIND_LABELS_FILENAME']
    split_filename = os.environ['PDBBIND_TEST_SPLIT_FILENAME']

    data_stats_df = get_data_stats(data_filename)

    print('\nTesting pdbbind feature generator')
    gen = dataset_generator(
        data_filename, split_filename, labels_filename, grid_config,
        shuffle=True, repeat=1, max_pdbs=10)

    for i, (pdbcode, feature, label) in enumerate(gen):
        print('Generating feature {:} {:} -> feature {:}, label {:}'.format(
            i, pdbcode, feature.shape, label.shape))
