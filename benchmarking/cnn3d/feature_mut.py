import math
import os

import dotenv as de
import numpy as np
import pandas as pd

de.load_dotenv(de.find_dotenv(usecwd=True))

import atom3d.shard.shard as sh

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
    'neg_to_pos_ratio': 1.0,
    # Max number of positive regions to take from a structure. -1 means all.
    'max_pos_per_shard': 200,
})


def __get_mutation_center(struct_df, label_info, center_at_mut=True):
    if center_at_mut:
        # Use CA position of the mutated residue as center for subgrid center
        sel = ((struct_df.chain == label_info.chain) &
               (struct_df.residue == label_info.residue) &
               (struct_df.name == 'CA'))
        mutation_pos = struct_df[sel][['x', 'y', 'z']].astype(np.float32)
        mutation_center = util.get_center(mutation_pos)
    else:
        pos = struct_df[['x', 'y', 'z']].astype(np.float32)
        mutation_center = util.get_center(pos)
    return mutation_center


def df_to_feature(struct_df, label_info, grid_config, center_at_mut=True,
                  random_seed=None):
    mutation_center = __get_mutation_center(struct_df, label_info, center_at_mut)

    rot_mat = subgrid_gen.gen_rot_matrix(grid_config, random_seed=random_seed)
    grid = subgrid_gen.get_grid(
        struct_df, mutation_center, config=grid_config, rot_mat=rot_mat)
    return grid


def read_all_labels(sharded):
    num_shards = sharded.get_num_shards()
    # Read all labels
    frames = []
    for shard_num in range(num_shards):
        df = sharded.read_shard(shard_num, key='labels')
        df['shard'] = shard_num
        frames.append(df)
    all_df = pd.concat(frames)
    neg_df = all_df[all_df.label==0].reset_index(drop=True)
    pos_df = all_df[all_df.label==1].reset_index(drop=True)
    return neg_df, pos_df


def __num_to_use(num_pos, num_neg, testing, grid_config):
    if grid_config.neg_to_pos_ratio == -1 or testing:
        num_pos_to_use, num_neg_to_use = num_pos, num_neg
    else:
        num_pos_to_use = min(num_pos, num_neg / grid_config.neg_to_pos_ratio)
        if (not testing) and (grid_config.max_pos_per_shard != -1):
            num_pos_to_use = min(
                num_pos_to_use, grid_config.max_pos_per_shard)
        num_neg_to_use = num_pos_to_use * grid_config.neg_to_pos_ratio
    num_pos_to_use = int(math.ceil(num_pos_to_use))
    num_neg_to_use = int(math.ceil(num_neg_to_use))
    return num_pos_to_use, num_neg_to_use


def dataset_generator(sharded, grid_config, shuffle=True, repeat=None,
                      add_flag=True, center_at_mut=True,
                      testing=False, random_seed=None):

    num_shards = sharded.get_num_shards()
    all_shard_nums = np.arange(num_shards)

    all_neg_labels_df, all_pos_labels_df = read_all_labels(sharded)

    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        if shuffle:
            p = np.random.permutation(len(all_shard_nums))
            all_shard_nums = all_shard_nums[p]

        for shard_num in all_shard_nums:
            neg_labels_df = all_neg_labels_df[all_neg_labels_df.shard==shard_num]
            pos_labels_df = all_pos_labels_df[all_pos_labels_df.shard==shard_num]

            if not testing and shuffle:
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

            shard_df = sharded.read_shard(shard_num, key='structures')

            for index, label_info in labels_df.iterrows():
                ensemble_name = label_info.ensemble
                ensemble_df = shard_df[shard_df.ensemble == ensemble_name]

                features = [] # Inmutated, mutated
                for is_mutated, subunit_name in enumerate(['original', 'mutated']):
                    struct_df = ensemble_df[ensemble_df.subunit == subunit_name]
                    fgrid = df_to_feature(struct_df, label_info, grid_config,
                                          center_at_mut, random_seed)
                    if add_flag:
                        flag = np.full(fgrid.shape[:-1] + (1,), is_mutated)
                        fgrid = np.concatenate([fgrid, flag], axis=3)
                    features.append(fgrid)

                yield ensemble_name, np.array(features), np.array([label_info.label])


def get_data_stats(sharded_list, center_at_mut=True):
    """
    Get the furthest distance from the protein's center and max residue ID for
    every protein in the sharded dataset.
    """
    data = []
    all_elements = []
    labels = []

    for i, sharded in enumerate(sharded_list):
        for shard_num, shard_df in sharded.iter_shards():
            labels_df = sharded.read_shard(shard_num, key='labels')

            for ensemble_name, ensemble_df in shard_df.groupby(['ensemble']):
                all_elements.extend(ensemble_df.element.values)
                label_info = labels_df[labels_df.ensemble == ensemble_name].squeeze()

                for subunit_name in ['original', 'mutated']:
                    struct_df = ensemble_df[ensemble_df.subunit == subunit_name]
                    pos = struct_df[['x', 'y', 'z']].astype(np.float32)
                    mutation_center = __get_mutation_center(
                        struct_df, label_info, center_at_mut)

                    max_dist = util.get_max_distance_from_center(pos, mutation_center)
                    num_atoms = struct_df.shape[0]
                    data.append((ensemble_name, subunit_name, max_dist, num_atoms))

                labels.append((i, shard_num, label_info.label))

    all_elements_df = pd.DataFrame(all_elements, columns=['element'])
    unique_elements = all_elements_df.element.unique()
    print('Unique elements ({:}): {:}'.format(len(unique_elements), unique_elements))
    print('\nElement counts:')
    print(all_elements_df.element.value_counts())
    print('\n')

    all_labels_df = pd.DataFrame(labels, columns=['sharded', 'shard_num', 'label'])
    print('\nLabel by dataset:')
    print(all_labels_df.groupby(['sharded', 'shard_num']).label.value_counts())
    print('\n')
    print(all_labels_df.label.value_counts())

    df = pd.DataFrame(data, columns=['ensemble', 'subunit', 'max_dist', 'num_atoms'])
    df = df.sort_values(by=['max_dist', 'num_atoms'],
                        ascending=[False, False]).reset_index(drop=True)
    print(df.describe())

    print(df[df.max_dist < 50].shape[0]*100.0/df.shape[0])
    return df


if __name__ == "__main__":
    sharded_path_list = [
        #os.environ['MUT_TRAIN_SHARDED'],
        #os.environ['MUT_VAL_SHARDED'],
        os.environ['MUT_TEST_SHARDED'],
        ]
    sharded_list = [sh.Sharded.load(path) for path in sharded_path_list]

    data_stats_df = get_data_stats(sharded_list, center_at_mut=True)

    print('\nTesting Mutation feature generator')
    gen = dataset_generator(
        sharded_list[0], grid_config, shuffle=True,
        repeat=1, add_flag=True, testing=False)

    for i, (ensemble, feature, label) in enumerate(gen):
        print('Generating feature {:} {:} -> feature {:}, score {:}'.format(
            i, ensemble, feature.shape, label))
