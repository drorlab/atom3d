import collections as col
import os

import dotenv as de
import numpy as np
import pandas as pd

de.load_dotenv(de.find_dotenv(usecwd=True))

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
})


def df_to_feature(struct_df, grid_config, random_seed=None):
    pos = struct_df[['x', 'y', 'z']].astype(np.float32)
    center = util.get_center(pos)

    rot_mat = subgrid_gen.gen_rot_matrix(grid_config, random_seed=random_seed)
    grid = subgrid_gen.get_grid(
        struct_df, center, config=grid_config, rot_mat=rot_mat)
    return grid


def dataset_generator(sharded, grid_config, score_type='gdt_ts',
                      shuffle=True, repeat=None, max_targets=None,
                      max_decoys=None, max_dist_threshold=300.0,
                      random_seed=None):

    all_target_names = np.squeeze(sharded.get_names())

    num_shards = sharded.get_num_shards()
    all_shard_nums = np.arange(num_shards)

    # Asume 1 shard per target
    assert len(all_shard_nums) == len(all_target_names)

    seen = col.defaultdict(set)

    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        if shuffle:
            p = np.random.permutation(len(all_shard_nums))
            all_shard_nums = all_shard_nums[p]
        shard_nums = all_shard_nums if max_targets is None else all_shard_nums[:max_targets]

        for i, shard_num in enumerate(shard_nums):
            scores_df = sharded.read_shard(shard_num, key='labels')
            target_df = sharded.read_shard(shard_num, key='structures')
            target_name = target_df.ensemble.unique()[0]

            decoy_names = target_df.subunit.unique()
            if len(seen[target_name]) == len(decoy_names):
                seen[target_name] = set()

            if shuffle:
                p = np.random.permutation(len(decoy_names))
                decoy_names = decoy_names[p]

            num_decoys = 0
            for j, decoy_name in enumerate(decoy_names):
                if (max_decoys is not None) and (num_decoys == max_decoys):
                    continue
                if decoy_name in seen[target_name]:
                    continue
                seen[target_name].add(decoy_name)

                struct_df = target_df[target_df.subunit == decoy_name]

                # Skip if max distance higher than threshold
                '''if max_dist_threshold is not None:
                    pos = struct_df[['x', 'y', 'z']].astype(np.float32)
                    max_dist = util.get_max_distance_from_center(
                        pos, util.get_center(pos))
                    if max_dist > max_dist_threshold:
                        #print('Skipping decoy {:}/{:} since max dist {:.2f} > {:.2f}'.format(
                        #    target_name, decoy_name, max_dist, max_dist_threshold))
                        continue'''

                feature = df_to_feature(struct_df, grid_config, random_seed)
                score = scores_df[(scores_df.ensemble == target_name) & \
                                  (scores_df.subunit == decoy_name)][score_type].values
                num_decoys += 1

                yield '{:}/{:}.pdb'.format(target_name, decoy_name), feature, score


def get_data_stats(sharded_list):
    """
    Get the furthest distance from the protein's center and max residue ID for
    every protein in the sharded dataset.
    """
    data = []
    for i, sharded in enumerate(sharded_list):
        for _, shard_df in sharded.iter_shards():
            for (target, decoy), struct_df in shard_df.groupby(['ensemble', 'subunit']):
                pos = struct_df[['x', 'y', 'z']].astype(np.float32)
                max_dist = util.get_max_distance_from_center(
                    pos, util.get_center(pos))
                max_res = struct_df.residue.max()
                data.append((i, target, decoy, max_dist, max_res))
    df = pd.DataFrame(data, columns=['sharded', 'target', 'decoy', 'max_dist', 'max_res'])
    df = df.sort_values(by=['sharded', 'max_dist', 'max_res'],
                        ascending=[True, False, False]).reset_index(drop=True)
    print(df.describe())

    print(df[df.max_dist < 90].shape[0]*100.0/df.shape[0])
    print(df[df.max_dist < 90].target.unique().shape[0]*100.0/float(df.target.unique().shape[0]))
    return df


if __name__ == "__main__":
    sharded_path_list = [
        #os.environ['PSP_TRAIN_SHARDED'],
        os.environ['PSP_VAL_SHARDED'],
        #os.environ['PSP_TEST_SHARDED'],
    ]
    sharded_list = [sh.load_sharded(path) for path in sharded_path_list]

    #data_stats_df = get_data_stats(sharded_list)

    print('\nTesting PSP feature generator')
    gen = dataset_generator(
        sharded_list[-1], grid_config, score_type='gdt_ts',
        shuffle=True, repeat=5, max_targets=None, max_decoys=1,
        max_dist_threshold=150.0)

    for i, (struct_name, feature, score) in enumerate(gen):
        print('Generating feature {:} {:} -> feature {:}, score {:}'.format(
            i, struct_name, feature.shape, score.shape))
