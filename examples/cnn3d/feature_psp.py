import click
import os
import subprocess
import tqdm
import warnings

import numpy as np
import pandas as pd

import dotenv as de
de.load_dotenv(de.find_dotenv())

import atom3d.psp.util as psp_util
import atom3d.util.datatypes as dt
import atom3d.util.shard as sh

import examples.cnn3d.load_data as load_data


def read_score_labels(scores_dir, targets):
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


def pdb_to_subgrid_dataset(pdb_filename, grid_size=24):
    """
    Map protein to subgrid feature.
    """
    map_filename = os.path.join(os.environ['TEMP_PATH'],
                                'map_' + str(os.getpid()) + '_pred.bin')
    try:
        subprocess.check_output(
            "{:} --mode map -i {:} --native -m {:} -v 0.8 -o {:}".format(
                os.environ['MAP_GENERATOR_PATH'], pdb_filename,
                grid_size, map_filename),
            universal_newlines=True,
            shell=True)
    except:
        warnings.warn('# Mapping failed, ignoring protein {:}'.format(pdb_filename))
        return None
    dataset = load_data.read_data_set(map_filename)
    os.remove(map_filename)
    return dataset


def df_to_subgrid_dataset(df, struct_name='', grid_size=24):
    """
    Given pandas DataFrame as input, convert it into pdb temporarily to generate
    the subgrid mapping (+ metadata) for each residue in the structure.
    """
    if struct_name == '' or struct_name is None:
        struct_name = str(os.getpid()) + '.pdb'

    pdb_filename = os.path.join(os.environ['TEMP_PATH'],
                                'pred_'+ struct_name.replace('/','__'))
    dt.write_pdb(pdb_filename, dt.df_to_bp(df))
    dataset = pdb_to_subgrid_dataset(pdb_filename, grid_size=grid_size)
    os.remove(pdb_filename)
    return dataset


def df_to_subgrid_feature(df, struct_name, grid_size):
    """
    Convert structure to subgrid mapping.
    """
    subgrid = df_to_subgrid_dataset(df, struct_name, grid_size)
    assert subgrid != None, "Failed to generate subgrid map for %s" % struct_name
    return subgrid.maps


def reshape_subgrid_map_channel_last(array, nb_type, grid_size):
    array = np.reshape(array, [-1, nb_type, grid_size, grid_size, grid_size])
    array = np.transpose(array, (0, 2, 3, 4, 1))
    return array


def subgrid_dataset_generator(sharded, scores_dir, score_type='gdt_ts',
                              nb_type=169, grid_size=24,
                              shuffle=True, random_seed=None, repeat=None,
                              max_targets=None, max_decoys=None,
                              res_count=None, min_res=100):
    """
    Generator that convert sharded HDF data structure to subgrid feature and
    also return the score labels. Skip target that has residue less than
    <min_res> if specified.
    """
    all_target_names = sh.get_names(sharded)
    scores_df = read_score_labels(scores_dir, all_target_names)

    np.random.seed(random_seed)
    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        #print('\nEpoch {:}'.format(epoch))
        if shuffle:
            p = np.random.permutation(len(all_target_names))
            all_target_names = all_target_names[p]
        target_names = all_target_names if max_targets is None else all_target_names[:max_targets]

        for i, target_name in enumerate(target_names):
            target_df = sh.read_ensemble(sharded, target_name)
            if (min_res is not None) and (target_df.residue.max() < min_res):
                print('Skipping target {:} since max res {:} < {:}'.format(
                    target_name, target_df.residue.max(), min_res))
                continue

            decoy_names = target_df.subunit.unique()
            if shuffle:
                p = np.random.permutation(len(decoy_names))
                decoy_names = decoy_names[p]
            if max_decoys is not  None:
                decoy_names = decoy_names[:max_decoys]

            for j, decoy_name in enumerate(decoy_names):
                struct_df = target_df[target_df.subunit == decoy_name]
                num_res = len(struct_df.residue.unique())

                if (min_res is not None) and (num_res < min_res):
                    print('Skipping decoy {:}/{:} since num res {:} < {:}'.format(
                        target_name, decoy_name, num_res, min_res))
                    continue

                feature = df_to_subgrid_feature(
                    struct_df, '{:}/{:}'.format(target_name, decoy_name),
                    grid_size)
                assert(len(feature) == num_res)
                score = scores_df[(scores_df.target == target_name) & \
                                  (scores_df.decoy == decoy_name)][score_type].values

                if num_res < min_res:
                    print("ERROR {:}/{:}".format(target_name, decoy_name))
                    import pdb; pdb.set_trace()
                    assert False
                if (res_count is not None) and (num_res > res_count):
                    index = np.random.choice(
                        np.arange(num_res), res_count, replace=False)
                    index = np.sort(index)
                    feature = feature[index]

                #feature = np.reshape(feature, [-1, grid_size, grid_size, grid_size])
                #feature = np.transpose(feature, (1, 2, 3, 0))
                feature = reshape_subgrid_map_channel_last(feature, nb_type, grid_size)

                #print('Target {:} ({:}/{:}): decoy {:} ({:}/{:}) -> feature {:}, score {:}'.format(
                #    target_name, i+1, len(target_names), decoy_name, j+1,
                #    len(decoy_names), feature.shape, score.shape))

                yield target_name, decoy_name, num_res, feature, score


def get_num_residues(sharded):
    """Get number of residues of all structures in the dataset."""
    num_residues = []
    for shard_df in sh.iter_shards(sharded):
        for key, struct_df in shard_df.groupby(['ensemble', 'subunit']):
            print('{:} -> {:} res'.format(key, struct_df.residue.max()))
            num_residues.append(struct_df.residue.max())
    return num_residues


if __name__ == "__main__":
    sharded = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_50/train_decoy_50@250'
    #sharded = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_50/val_decoy_50@25'
    #sharded = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_50/test_decoy_20@20'
    scores_dir = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/labels/scores'

    #num_residues = get_num_residues(sharded)
    #print('\nMAX residues: {:}'.format(np.max(num_residues)))
    #print('MIN residues: {:}'.format(np.min(num_residues)))

    print('Testing feature generator')
    gen = subgrid_dataset_generator(
        sharded, scores_dir, score_type='gdt_ts', nb_type=169, grid_size=24,
        shuffle=True, random_seed=13, repeat=1,
        max_targets=10, max_decoys=10, res_count=150, min_res=150)

    for i, (target_name, decoy_name, num_res, feature, score) in enumerate(gen):
        print('Generating feature {:} {:}/{:} -> res {:}, feature {:}, score {:}'.format(
            i, target_name, decoy_name, num_res, feature.shape, score.shape))
