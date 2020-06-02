import math
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
        'H': 0,
        'C': 1,
        'O': 2,
        'N': 3,
        'S': 4,
        'CL': 5,
        'F': 6,
    },
    # Radius of the grids to generate, in angstroms.
    'radius': 25.0,
    # Resolution of each voxel, in angstroms.
    'resolution': 1.0,
    # Number of directions to apply for data augmentation.
    'num_directions': 20,
    # Number of rolls to apply for data augmentation.
    'num_rolls': 20,
    # Number of negatives to sample per positive example. -1 means all.
    # positive = A (active), negative = I (inactive)
    'neg_to_pos_ratio': 1,
})


def __get_subunit_name(subunits, mode='inactive'):
    assert len(subunits) == 2
    for name in subunits:
        if name.endswith('_' + mode):
            return name
    return ''


def df_to_feature(struct_df, grid_config, center_around_Cs, random_seed=None):
    # Consider only atoms that have mapping for computing center.
    # If <center_around_Cs> is set, consider only carbon atoms.
    if center_around_Cs:
        pruned_struct_df = struct_df[struct_df.element == 'C']
    else:
        pruned_struct_df = struct_df[struct_df.element.isin(grid_config.element_mapping.keys())]

    pos = pruned_struct_df[['x', 'y', 'z']].astype(np.float32)
    # Use center of ligand for subgrid center
    ligand_pos = pruned_struct_df[pruned_struct_df.chain == 'L'][['x', 'y', 'z']].astype(
        np.float32)
    ligand_center = util.get_center(ligand_pos)

    rot_mat = subgrid_gen.gen_rot_matrix(grid_config, random_seed=random_seed)
    grid = subgrid_gen.get_grid(
        struct_df, ligand_center, config=grid_config, rot_mat=rot_mat)
    return grid


def read_all_labels(sharded):
    num_shards = sh.get_num_shards(sharded)
    # Read all labels
    frames = []
    for shard_num in range(num_shards):
        df = sh.read_shard(sharded, shard_num, key='labels')
        df['shard'] = shard_num
        frames.append(df)
    all_df = pd.concat(frames)
    neg_df = all_df[all_df.label=='I'].reset_index(drop=True)
    pos_df = all_df[all_df.label=='A'].reset_index(drop=True)
    return neg_df, pos_df


def __num_to_use(num_pos, num_neg, testing, grid_config):
    if grid_config.neg_to_pos_ratio == -1 or testing:
        num_pos_to_use, num_neg_to_use = num_pos, num_neg
    else:
        num_pos_to_use = min(num_pos, num_neg / grid_config.neg_to_pos_ratio)
        num_neg_to_use = num_pos_to_use * grid_config.neg_to_pos_ratio
    num_pos_to_use = int(math.ceil(num_pos_to_use))
    num_neg_to_use = int(math.ceil(num_neg_to_use))
    return num_pos_to_use, num_neg_to_use


def dataset_generator(sharded, grid_config, shuffle=True, repeat=None,
                      max_shards=None, add_flag=True, testing=False,
                      center_around_Cs=True, random_seed=None):

    #np.random.seed(random_seed)

    num_shards = sh.get_num_shards(sharded)
    all_shard_nums = np.arange(num_shards)

    all_neg_labels_df, all_pos_labels_df = read_all_labels(sharded)

    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        if shuffle:
            p = np.random.permutation(len(all_shard_nums))
            all_shard_nums = all_shard_nums[p]
        shard_nums = all_shard_nums if max_shards is None else all_shard_nums[:max_shards]

        for shard_num in all_shard_nums:
            neg_labels_df = all_neg_labels_df[all_neg_labels_df.shard==shard_num]
            pos_labels_df = all_pos_labels_df[all_pos_labels_df.shard==shard_num]

            if shuffle:
                neg_labels_df = neg_labels_df.sample(frac=1).reset_index(drop=True)
                pos_labels_df = pos_labels_df.sample(frac=1).reset_index(drop=True)

            # Sample pos and neg samples
            num_pos_to_use, num_neg_to_use = __num_to_use(
                pos_labels_df.shape[0], neg_labels_df.shape[0], testing, grid_config)

            if pos_labels_df.shape[0] == num_pos_to_use:
                pos_labels_df = pos_labels_df.reset_index(drop=True)
            else:
                pos_labels_df = pos_labels_df.sample(num_pos_to_use, replace=True).reset_index(drop=True)
            if neg_labels_df.shape[0] == num_neg_to_use:
                neg_labels_df = neg_labels_df.reset_index(drop=True)
            else:
                neg_labels_df = neg_labels_df.sample(num_neg_to_use, replace=True).reset_index(drop=True)

            labels_df = pd.DataFrame(
                list(util.intersperse(pos_labels_df.values, neg_labels_df.values)),
                columns=pos_labels_df.columns.values)

            shard_df = sh.read_shard(sharded, shard_num, key='structures')

            for index, label_info in labels_df.iterrows():
                ensemble_name = label_info.ensemble
                ensemble_df = shard_df[shard_df.ensemble == ensemble_name]
                subunits = ensemble_df.subunit.unique()
                inactive = __get_subunit_name(subunits, mode='inactive')
                active = __get_subunit_name(subunits, mode='active')

                features = [] # Inactive, active
                for is_active, subunit_name in enumerate([inactive, active]):
                    struct_df = ensemble_df[ensemble_df.subunit == subunit_name]
                    fgrid = df_to_feature(struct_df, grid_config,
                                          center_around_Cs, random_seed)
                    if add_flag:
                        flag = np.full(fgrid.shape[:-1] + (1,), is_active)
                        fgrid = np.concatenate([fgrid, flag], axis=3)
                    features.append(fgrid)

                yield ensemble_name, np.array(features), np.array([label_info.label == 'A'], dtype=np.int8)


def get_data_stats(sharded_list):
    """
    Get the furthest distance from the protein's center and max residue ID for
    every protein in the sharded dataset.
    """
    data = []
    all_elements = []
    labels = []
    for i, sharded in enumerate(sharded_list):
        for shard_num, shard_df in sh.iter_shards(sharded):
            labels_df = sh.read_shard(sharded, shard_num, key='labels')

            for ensemble, ensemble_df in shard_df.groupby(['ensemble']):
                all_elements.extend(ensemble_df.element.values)

                subunits = ensemble_df.subunit.unique()
                inactive = __get_subunit_name(subunits, mode='inactive')
                active = __get_subunit_name(subunits, mode='active')

                for subunit_name in [inactive, active]:
                    struct_df = ensemble_df[ensemble_df.subunit == subunit_name]
                    pos = struct_df[['x', 'y', 'z']].astype(np.float32)
                    ligand_pos = struct_df[struct_df.chain == 'L'][['x', 'y', 'z']].astype(
                        np.float32)
                    ligand_center = util.get_center(ligand_pos)

                    max_dist = util.get_max_distance_from_center(pos, ligand_center)
                    num_atoms = struct_df.shape[0]
                    data.append((ensemble, subunit_name, max_dist, num_atoms))

                labels.append((i, shard_num, labels_df[labels_df.ensemble == ensemble].label.values[0] == 'A'))

                    #print('{:}/{:} -> max dist: {:.2f}'.format(
                    #    ensemble, subunit_name, max_dist))

    all_elements_df = pd.DataFrame(all_elements, columns=['element'])
    unique_elements = all_elements_df.element.unique()
    print('Unique elements ({:}): {:}'.format(len(unique_elements), unique_elements))
    print('\nElement counts:')
    print(all_elements_df.element.value_counts())
    print('\n')

    all_labels_df = pd.DataFrame(labels, columns=['sharded', 'shard_num', 'label'])
    print('\nLabel dist by dataset:')
    print(all_labels_df.groupby(['sharded', 'shard_num']).label.value_counts())
    print('\n')

    df = pd.DataFrame(data, columns=['ensemble', 'subunit', 'max_dist', 'num_atoms'])
    df = df.sort_values(by=['max_dist', 'num_atoms'],
                        ascending=[False, False]).reset_index(drop=True)
    print(df.describe())

    print(df[df.max_dist < 90].shape[0]*100.0/df.shape[0])
    import pdb; pdb.set_trace()
    return df


if __name__ == "__main__":
    sharded_list = [
        '/oak/stanford/groups/rondror/projects/atom3d/ligand_activity_prediction/split-20200524/pairs_train@10',
        '/oak/stanford/groups/rondror/projects/atom3d/ligand_activity_prediction/split-20200524/pairs_val@10',
        '/oak/stanford/groups/rondror/projects/atom3d/ligand_activity_prediction/split-20200524/pairs_test@10']

    data_stats_df = get_data_stats(sharded_list)

    print('Testing Activity feature generator')
    gen = dataset_generator(sharded_list[0], grid_config, shuffle=True,
                            repeat=1, max_shards=10, add_flag=True)

    for i, (ensemble, feature, label) in enumerate(gen):
        print('Generating feature {:} {:} -> feature {:}, score {:}'.format(
            i, ensemble, feature.shape, label))
