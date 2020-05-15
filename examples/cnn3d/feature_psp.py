import os
import subprocess
import tqdm
import warnings

import numpy as np
import pandas as pd

import dotenv as de
de.load_dotenv(de.find_dotenv())

import atom3d.util.datatypes as dt
import atom3d.util.shard as sh

import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util


grid_config = util.dotdict({
    # Mapping from elements to position in channel dimension.
    'element_mapping': {
        'C': 0,
        'O': 1,
        'N': 2,
        'S': 3,
    },
    # Radius of the grids to generate, in angstroms.
    'radius': 50.0,
    # Resolution of each voxel, in angstroms.
    'resolution': 1.0,
    # Number of directions to apply for data augmentation.
    'num_directions': 20,
    # Number of rolls to apply for data augmentation.
    'num_rolls': 20,
    # Random seed
    'random_seed': 131313,
})


def read_scores(scores_dir, targets):
    """
    Return a pandas DataFrame containing scores of all decoys for all targets
    in <targets>. Search in <scores_dir> for the label files.
    """
    frames = []
    for target in targets:
        df = pd.read_csv(os.path.join(scores_dir, '{:}.dat'.format(target)),
                         delimiter='\s+', engine='python').dropna()
        frames.append(df)
    scores_df = dt.merge_dfs(frames)
    return scores_df


def df_to_feature(struct_df, grid_config):
    pos = struct_df[['x', 'y', 'z']].astype(np.float32)
    center = util.get_center(pos)

    rot_mat = subgrid_gen.gen_rot_matrix(grid_config)
    grid = subgrid_gen.get_grid(
        struct_df, center, config=grid_config, rot_mat=rot_mat)
    return grid


def dataset_generator(sharded, scores_dir, grid_config, score_type='gdt_ts',
                      shuffle=True, repeat=None, max_targets=None,
                      max_decoys=None, max_dist_threshold=150.0):
    """
    Generator that convert sharded HDF dataset to grid features and
    also return the score labels. Skip structure with max distance above
    <max_dist_threshold> if specified.
    """
    all_target_names = sh.get_names(sharded)
    scores_df = read_scores(scores_dir, all_target_names)

    np.random.seed(grid_config.random_seed)
    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        if shuffle:
            p = np.random.permutation(len(all_target_names))
            all_target_names = all_target_names[p]
        target_names = all_target_names if max_targets is None else all_target_names[:max_targets]

        for i, target_name in enumerate(target_names):
            target_df = sh.read_ensemble(sharded, target_name)

            decoy_names = target_df.subunit.unique()
            if shuffle:
                p = np.random.permutation(len(decoy_names))
                decoy_names = decoy_names[p]
            if max_decoys is not None:
                decoy_names = decoy_names[:max_decoys]

            for j, decoy_name in enumerate(decoy_names):
                struct_df = target_df[target_df.subunit == decoy_name]

                # Skip if max distance higher than threshold
                '''if max_dist_threshold is not None:
                    pos = struct_df[['x', 'y', 'z']].astype(np.float32)
                    max_dist = util.get_max_distance_from_center(
                        pos, util.get_center(pos))
                    if max_dist > max_dist_threshold:
                        print('Skipping decoy {:}/{:} since max dist {:.2f} > {:.2f}'.format(
                            target_name, decoy_name, max_dist, max_dist_threshold))
                        continue'''

                feature = df_to_feature(struct_df, grid_config)
                score = scores_df[(scores_df.target == target_name) & \
                                  (scores_df.decoy == decoy_name)][score_type].values

                #print('Target {:} ({:}/{:}): decoy {:} ({:}/{:}) -> feature {:}, score {:}'.format(
                #    target_name, i+1, len(target_names), decoy_name, j+1,
                #    len(decoy_names), feature.shape, score.shape))

                yield '{:}/{:}.pdb'.format(target_name, decoy_name), feature, score


def get_data_stats(sharded):
    """
    Get the furthest distance from the protein's center and max residue ID for
    every protein in the sharded dataset.
    """
    data = []
    for shard_df in sh.iter_shards(sharded):
        for (target, decoy), struct_df in shard_df.groupby(['ensemble', 'subunit']):
            pos = struct_df[['x', 'y', 'z']].astype(np.float32)
            max_dist = subgrid_gen.get_max_distance_from_center(
                pos, subgrid_gen.get_center(pos))
            max_res = struct_df.residue.max()
            data.append((target, decoy, max_dist, max_res))
            #print('{:}/{:} -> max dist: {:.2f}, max res: {:}'.format(
            #    target, decoy, max_dist, max_res))
    df = pd.DataFrame(data, columns=['target', 'decoy', 'max_dist', 'max_res'])
    df = df.sort_values(by=['max_dist', 'max_res'],
                        ascending=[False, False]).reset_index(drop=True)
    print(df.describe())

    print(df[df.max_dist < 90].shape[0]*100.0/df.shape[0])
    print(df[df.max_dist < 90].target.unique().shape[0]*100.0/float(df.target.unique().shape[0]))
    return df


if __name__ == "__main__":
    #sharded='/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/hdf/casp_scwrl@709'
    #sharded = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_50/train_decoy_50@250'
    sharded = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_50/val_decoy_50@25'
    #sharded = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_50/test_decoy_all@85'
    scores_dir = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/labels/scores'

    #data_stats_df = get_data_stats(sharded)

    print('Testing PSP feature generator')
    gen = dataset_generator(
        sharded, scores_dir, grid_config, score_type='gdt_ts', shuffle=True,
        repeat=1, max_targets=10, max_decoys=10, max_dist_threshold=150.0)

    for i, (struct_name, feature, score) in enumerate(gen):
        print('Generating feature {:} {:} -> feature {:}, score {:}'.format(
            i, struct_name, feature.shape, score.shape))
