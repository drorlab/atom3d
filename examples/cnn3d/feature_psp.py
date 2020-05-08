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


def read_score_labels(scores_dir, structures, score_type='gdt_ts'):
    """
    Return a pandas DataFrame containing scores of all structures in <structures>
    (<target_name>/<decoy_name>.pdb). Search in <scores_dir> for the
    label files.
    """
    scores = []
    for name in structures:
        target = psp_util.get_target_name(name)
        decoy = psp_util.get_decoy_name(name)
        df = pd.read_csv(os.path.join(scores_dir, '{:}.dat'.format(target)),
                         delimiter='\s+', engine='python').dropna()
        score = df[(df.target==target) & (df.decoy==decoy)][score_type].values[0]
        scores.append([name, score])
    return pd.DataFrame(scores, columns=['structure', 'score'])


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


def shard_to_subgrid_feature(sharded, struct_name, nb_type, grid_size):
    """
    Lookup struct_name (<target_name>/<decoy_name>.pdb) in sharded HDF5,
    and return its subgrid mapping (+ metadata) for each residue in the
    structure.
    """
    df = sh.read_structure(sharded, struct_name)
    subgrid = df_to_subgrid_dataset(df, struct_name, grid_size)
    assert subgrid != None, "Failed to generate subgrid map for %s" % struct_name
    feature = reshape_subgrid_map_channel_last(
        subgrid.maps, nb_type=nb_type, grid_size=grid_size)
    return feature


def reshape_subgrid_map_channel_last(array, nb_type, grid_size):
    array = np.reshape(array, [-1, nb_type, grid_size, grid_size, grid_size])
    array = np.transpose(array, (0, 2, 3, 4, 1))
    return array


def subgrid_dataset_generator(sharded, scores_dir, score_type='gdt_ts',
                              nb_type=169, grid_size=24,
                              shuffle=True, random_seed=None, repeat=None,
                              num_iters=None, max_res=None):
    """
    Generator that convert sharded HDF data structure to subgrid feature and
    also return the score labels.
    """
    all_struct_names = sh.get_names(sharded)
    scores_df = read_score_labels(scores_dir, all_struct_names,
                                  score_type=score_type)

    np.random.seed(random_seed)
    if repeat == None:
        repeat = 1
    for epoch in range(repeat):
        #print('\nEpoch {:}'.format(epoch))
        if shuffle:
            p = np.random.permutation(len(all_struct_names))
            all_struct_names = all_struct_names[p]
        struct_names = all_struct_names if num_iters is None else all_struct_names[:num_iters]
        for i, struct_name in enumerate(struct_names):
            feature = shard_to_subgrid_feature(sharded, struct_name, nb_type, grid_size)
            score = scores_df[scores_df.structure == struct_name].score.values

            if max_res is not None:
                if shuffle:
                    p = np.random.permutation(len(feature))
                    feature = feature[p]
                feature = feature[:max_res]

            #print('Reading structure {:} {:}/{:} -> feature {:}, score {:}'.format(
            #    struct_name, i+1, len(struct_names), feature.shape, score.shape))
            yield struct_name, feature, score


if __name__ == "__main__":
    sharded = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/split_hdf/decoy_20/val_decoy_20@10'
    scores_dir = '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_prediction/casp/labels/scores'

    print('Testing feature generator')
    gen = subgrid_dataset_generator(
        sharded, scores_dir, score_type='gdt_ts', nb_type=169, grid_size=24,
        shuffle=True, random_seed=13, repeat=3, num_iters=3, max_res=200)
    for struct_name, feature, s in gen:
        pass
