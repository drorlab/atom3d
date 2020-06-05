import os
import subprocess
import tqdm
import warnings

import collections as col
import numpy as np
import pandas as pd

import dotenv as de
de.load_dotenv(de.find_dotenv(usecwd=True))

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
        'P': 3,
    },
    # Radius of the grids to generate, in angstroms.
    'radius': 50.0,
    # Resolution of each voxel, in angstroms.
    'resolution': 1.0,
    # Number of directions to apply for data augmentation.
    'num_directions': 20,
    # Number of rolls to apply for data augmentation.
    'num_rolls': 20,
})


def df_to_feature(struct_df, grid_config, random_seed):
    pos = struct_df[['x', 'y', 'z']].astype(np.float32)
    center = util.get_center(pos)

    rot_mat = subgrid_gen.gen_rot_matrix(grid_config, random_seed=random_seed)
    grid = subgrid_gen.get_grid(
        struct_df, center, config=grid_config, rot_mat=rot_mat)
    return grid


def dataset_generator(sharded,  grid_config, shuffle=True, repeat=None,
                      max_shards=None, max_decoys=None, random_seed=None):

    num_shards = sharded.get_num_shards()
    all_shard_nums = np.arange(num_shards)

    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        if shuffle:
            p = np.random.permutation(len(all_shard_nums))
            all_shard_nums = all_shard_nums[p]
        shard_nums = all_shard_nums if max_shards is None else all_shard_nums[:max_shards]

        for i, shard_num in enumerate(shard_nums):
            labels_df = sharded.read_shard(shard_num, key='labels')
            shard_df = sharded.read_shard(shard_num, key='structures')

            all_structures = np.unique(
                shard_df[['ensemble', 'subunit']].values.astype(str), axis=0)
            if shuffle:
                p = np.random.permutation(len(all_structures))
                all_structures = all_structures[p]
            structures = all_structures if max_decoys is None else all_structures[:max_decoys]

            for j, (target_name, decoy_name) in enumerate(structures):
                label = labels_df[(labels_df.ensemble == target_name) &
                                  (labels_df.subunit == decoy_name)].label.values
                struct_df = shard_df[(shard_df.ensemble == target_name) &
                                     (shard_df.subunit == decoy_name)]
                feature = df_to_feature(struct_df, grid_config, random_seed)
                yield '{:}/{:}.pdb'.format(target_name, decoy_name), feature, label


def get_data_stats(sharded_list):
    data = []
    for sharded in sharded_list:
        for _, shard_df in sharded.iter_shards():
            for (target, decoy), struct_df in shard_df.groupby(['ensemble', 'subunit']):
                pos = struct_df[['x', 'y', 'z']].astype(np.float32)
                max_dist = util.get_max_distance_from_center(
                    pos, util.get_center(pos))
                max_res = struct_df.residue.max()
                data.append((target, decoy, max_dist, max_res))
    df = pd.DataFrame(data, columns=['target', 'decoy', 'max_dist', 'max_res'])
    df = df.sort_values(by=['max_dist', 'max_res'],
                        ascending=[False, False]).reset_index(drop=True)
    print(df.describe())

    print(df[df.max_dist < 50].shape[0]*100.0/df.shape[0])
    print(df[df.max_dist < 50].target.unique().shape[0]*100.0/float(df.target.unique().shape[0]))
    return df


if __name__ == "__main__":
    sharded_path_list = [
        #os.environ['RNA_TRAIN_SHARDED'],
        #os.environ['RNA_VAL_SHARDED'],
        os.environ['RNA_TEST_SHARDED']
    ]
    sharded_list = [sh.load_sharded(path) for path in sharded_path_list]

    data_stats_df = get_data_stats(sharded_list)

    print('\nTesting RNA feature generator')
    gen = dataset_generator(sharded_list[0], grid_config, shuffle=True,
                            repeat=1, max_shards=None, max_decoys=None)

    for i, (struct_name, feature, score) in enumerate(gen):
        print('Generating feature {:} {:} -> feature {:}, score {:}'.format(
            i, struct_name, feature.shape, score))
