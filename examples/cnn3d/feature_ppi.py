import collections as col
import math
import os

import dotenv as de
import numpy as np
import pandas as pd

de.load_dotenv(de.find_dotenv(usecwd=True))

import atom3d.ppi.neighbors as nb
import atom3d.util.shard as sh

import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util


grid_config = util.dotdict({
    # Mapping from elements to position in channel dimension.
    'element_mapping': {
        'C': 0,
        'O': 1,
        'N': 2,
        'S': 3
    },
    # Radius of the grids to generate, in angstroms.
    'radius': 17.0,
    # Resolution of each voxel, in angstroms.
    'resolution': 1.0,
    # Number of directions to apply for data augmentation.
    'num_directions': 20,
    # Number of rolls to apply for data augmentation.
    'num_rolls': 20,

    ### PPI specific
    # Number of negatives to sample per positive example. -1 means all.
    'neg_to_pos_ratio': 1,
    'neg_to_pos_ratio_testing': 1,
    # Max number of positive regions to take from a structure. -1 means all.
    'max_pos_regions_per_ensemble': 5,
    'max_pos_regions_per_ensemble_testing': 5,
    # Whether to use all negative at test time.
    'full_test': False,
})


def df_to_feature(struct0, struct1, center0, center1, grid_config,
                  random_seed=None):
    def __feature(struct, center):
        rot_mat = subgrid_gen.gen_rot_matrix(grid_config, random_seed=random_seed)
        grid = subgrid_gen.get_grid(
            struct, center, config=grid_config, rot_mat=rot_mat)
        return grid

    grid0 = __feature(struct0, center0)
    grid1 = __feature(struct1, center1)
    feature = np.array([grid0, grid1])
    return feature


def __num_to_use(num_pos, num_neg, testing, grid_config):
    if testing:
        neg_to_pos_ratio = grid_config.neg_to_pos_ratio_testing
        max_pos_regions_per_ensemble = grid_config.max_pos_regions_per_ensemble_testing
    else:
        neg_to_pos_ratio = grid_config.neg_to_pos_ratio
        max_pos_regions_per_ensemble = grid_config.max_pos_regions_per_ensemble

    if neg_to_pos_ratio == -1 or (testing and grid_config.full_test):
        num_pos_to_use, num_neg_to_use = num_pos, num_neg
    else:
        num_pos_to_use = min(num_pos, num_neg / neg_to_pos_ratio)
        if (max_pos_regions_per_ensemble != -1):
            num_pos_to_use = min(num_pos_to_use, max_pos_regions_per_ensemble)
        num_neg_to_use = num_pos_to_use * neg_to_pos_ratio
    num_pos_to_use = int(math.ceil(num_pos_to_use))
    num_neg_to_use = int(math.ceil(num_neg_to_use))
    return num_pos_to_use, num_neg_to_use


def __get_res_pair_ca_coords(samples_df, structs_df):
    def __get_ca_coord(struct, res):
        coord = struct[(struct.residue == res) & (struct.name == 'CA')][['x', 'y', 'z']].values[0]
        return coord

    res_pairs = samples_df[['residue0', 'residue1']].values
    cas = []
    for (res0, res1) in res_pairs:
        try:
            coord0 = __get_ca_coord(structs_df[0], res0)
            coord1 = __get_ca_coord(structs_df[1], res1)
            cas.append((res0, res1, coord0, coord1))
        except:
            pass
    return cas


def dataset_generator(sharded, grid_config, shuffle=True, repeat=None,
                      max_num_ensembles=None, testing=False,
                      use_shard_nums=None, random_seed=None):

    if use_shard_nums is None:
        num_shards = sharded.get_num_shards()
        all_shard_nums = np.arange(num_shards)
    else:
        all_shard_nums = use_shard_nums

    seen = col.defaultdict(set)
    ensemble_count = 0

    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        if shuffle:
            p = np.random.permutation(len(all_shard_nums))
            all_shard_nums = all_shard_nums[p]
        shard_nums = all_shard_nums

        for i, shard_num in enumerate(shard_nums):
            shard_neighbors_df = sharded.read_shard(shard_num, key='neighbors')
            shard_structs_df = sharded.read_shard(shard_num, key='structures')

            ensemble_names = shard_structs_df.ensemble.unique()
            if len(seen[shard_num]) == len(ensemble_names):
                seen[shard_num] = set()

            if shuffle:
                p = np.random.permutation(len(ensemble_names))
                ensemble_names = ensemble_names[p]

            for j, ensemble_name in enumerate(ensemble_names):
                if (max_num_ensembles != None) and (ensemble_count >= max_num_ensembles):
                    return
                if ensemble_name in seen[shard_num]:
                    continue
                seen[shard_num].add(ensemble_name)
                ensemble_count += 1

                ensemble_df = shard_structs_df[shard_structs_df.ensemble == ensemble_name]
                # Subunits
                names, (bdf0, bdf1, udf0, udf1) = nb.get_subunits(ensemble_df)
                structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
                # Get positives
                pos_neighbors_df = shard_neighbors_df[shard_neighbors_df.ensemble0 == ensemble_name]
                # Get negatives
                neg_neighbors_df = nb.get_negatives(pos_neighbors_df, structs_df[0], structs_df[1])

                # Throw away non empty hetero/insertion_code
                non_heteros = []
                for df in structs_df:
                    non_heteros.append(df[(df.hetero==' ') & (df.insertion_code==' ')].residue.unique())
                pos_neighbors_df = pos_neighbors_df[pos_neighbors_df.residue0.isin(non_heteros[0]) & \
                                                    pos_neighbors_df.residue1.isin(non_heteros[1])]
                neg_neighbors_df = neg_neighbors_df[neg_neighbors_df.residue0.isin(non_heteros[0]) & \
                                                    neg_neighbors_df.residue1.isin(non_heteros[1])]

                # Sample pos and neg samples
                num_pos = pos_neighbors_df.shape[0]
                num_neg = neg_neighbors_df.shape[0]
                num_pos_to_use, num_neg_to_use = __num_to_use(num_pos, num_neg, testing, grid_config)

                if shuffle:
                    pos_neighbors_df = pos_neighbors_df.sample(frac=1).reset_index(drop=True)
                    neg_neighbors_df = neg_neighbors_df.sample(frac=1).reset_index(drop=True)
                if pos_neighbors_df.shape[0] == num_pos_to_use:
                    pos_samples_df = pos_neighbors_df.reset_index(drop=True)
                else:
                    pos_samples_df = pos_neighbors_df.sample(num_pos_to_use, replace=True).reset_index(drop=True)
                if neg_neighbors_df.shape[0] == num_neg_to_use:
                    neg_samples_df = neg_neighbors_df.reset_index(drop=True)
                else:
                    neg_samples_df = neg_neighbors_df.sample(num_neg_to_use, replace=True).reset_index(drop=True)

                pos_pairs_cas = __get_res_pair_ca_coords(pos_samples_df, structs_df)
                neg_pairs_cas = __get_res_pair_ca_coords(neg_samples_df, structs_df)

                pos_features = []
                for (res0, res1, center0, center1) in pos_pairs_cas:
                    fgrid = df_to_feature(structs_df[0], structs_df[1], center0,
                                          center1, grid_config, random_seed)
                    pos_features.append(('{:}/{:}/{:}'.format(ensemble_name, res0, res1), fgrid, np.array([1])))

                neg_features = []
                for (res0, res1, center0, center1) in neg_pairs_cas:
                    fgrid = df_to_feature(structs_df[0], structs_df[1], center0,
                                          center1, grid_config, random_seed)
                    neg_features.append(('{:}/{:}/{:}'.format(ensemble_name, res0, res1), fgrid, np.array([0])))

                for f in util.intersperse(pos_features, neg_features):
                    yield f


def get_data_stats(sharded_list):
    data = []
    for i, sharded in enumerate(sharded_list):
        num_shards = sharded.get_num_shards()
        for _, shard_num in enumerate(range(num_shards)):
            shard_neighbors_df = sharded.read_shard(shard_num, key='neighbors')
            shard_structs_df = sharded.read_shard(shard_num, key='structures')

            ensemble_names = shard_structs_df.ensemble.unique()
            for _, ensemble_name in enumerate(ensemble_names):
                ensemble_df = shard_structs_df[shard_structs_df.ensemble == ensemble_name]
                # Subunits
                names, (bdf0, bdf1, udf0, udf1) = nb.get_subunits(ensemble_df)
                structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
                # Get positives
                pos_neighbors_df = shard_neighbors_df[shard_neighbors_df.ensemble0 == ensemble_name]
                num_pos = pos_neighbors_df.shape[0]
                data.append((i, shard_num, ensemble_name, num_pos))

    df = pd.DataFrame(data, columns=['sharded', 'shard_num', 'ensemble', 'num_pos'])
    df = df.sort_values(['sharded', 'num_pos'], ascending=[True, False]).reset_index(drop=True)
    print(df.describe())
    return df


if __name__ == "__main__":
    sharded_path_list = [
        #os.environ['PPI_TRAIN_SHARDED'],
        #os.environ['PPI_VAL_SHARDED'],
        os.environ['PPI_TEST_SHARDED'],
        ]
    sharded_list = [sh.load_sharded(path) for path in sharded_path_list]

    stats_df = get_data_stats(sharded_list)

    print('\nTesting PPI feature generator')
    num_repeat = 3
    gen = dataset_generator(
        sharded_list[0], grid_config, shuffle=True, repeat=num_repeat,
        max_num_ensembles=15, testing=False,
        use_shard_nums=None, random_seed=12345)

    data = []
    for i, (ensemble, feature, label) in enumerate(gen):
        print('Generating feature {:} {:} -> pos features {:}, neg features {:}'.format(
            i, ensemble, feature.shape, label.shape))
        data.append((i, ensemble))
    df = pd.DataFrame(data, columns=['index', 'structure'])
    df[['ensemble', 'res0', 'res1']] = df.structure.str.split('/', expand=True)
    print('Num of unique ensembles: {:}'.format(len(df.ensemble.unique())))
