import os
import random

import numpy as np
import pandas as pd
import parallel as par
import scipy.spatial
import torch
from atom3d.util import *

# import atom3d.util.datatypes as dt
import atom3d.shard.shard as sh
import atom3d.shard.shard_ops as sho

seed = 131313
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

allowed_atoms = ['C', 'O', 'N', 'S', 'P', 'SE'] #'ZN', 'NA', 'FE', 'CA', 'MG', 'CU', 'CL', 'F'] 


res_wt_dict={}
with open('train_res_weights') as f:
    for line in f:
        res, wt = line.strip().split()
        res_wt_dict[res] = float(wt)




def shard_envs(input_path, output_path, num_threads=8, subsample=True):
    input_sharded = sh.Sharded.load(input_path)
    keys = input_sharded.get_keys()
    if keys != ['ensemble']:
        raise RuntimeError('Can only apply to sharded by ensemble.')
    output_sharded = sh.Sharded(output_path, keys)
    input_num_shards = input_sharded.get_num_shards()

    tmp_path = output_sharded.get_prefix() + f'_tmp@{input_num_shards:}'
    tmp_sharded = sh.Sharded(tmp_path, keys)

    not_written = []
    for i in range(input_num_shards):
        shard = output_sharded._get_shard(i)
        if not os.path.exists(shard):
            not_written.append(i)

    print(f'Using {num_threads:} threads')

    inputs = [(input_sharded, tmp_sharded, shard_num, subsample)
              for shard_num in range(8)]

    # with multiprocessing.Pool(processes=num_threads) as pool:
    #     pool.starmap(_shard_envs, inputs)
    par.submit_jobs(_shard_envs, inputs, num_threads)

    sho.reshard(tmp_sharded, output_sharded)
    tmp_sharded.delete_files()


def _shard_envs(input_sharded, output_sharded, shard_num, subsample):
    print(f'Processing shard {shard_num:}')
    shard = input_sharded.read_shard(shard_num)
    num_structures = len(shard['ensemble'].unique())

    envs = []
    for structure, x in shard.groupby('ensemble'):

        if len(x['subunit'].unique()) > 1:
            raise RuntimeError('Cannot find pairs on existing ensemble')
        # Only keep first model.
        x = x[x['model'] == sorted(x['model'].unique())[0]]
        subunits = _gen_subunits(x, subsample)
        if not subunits:
            continue
        envs.append(pd.concat(subunits))
        
    envs = pd.concat(envs).reset_index(drop=True)
    num_envs = len(envs['subunit'].unique())
    output_sharded._write_shard(shard_num, envs)
    print(f'Done processing shard {shard_num:}, generated {num_envs:} '
                f'environments from {num_structures:} structures.')


def _gen_subunits(df, subsample):
    """Extract environments as subunits """
    subunits = []
    # df = df.set_index(['chain', 'residue', 'resname'], drop=False)
    df = df.dropna(subset=['x','y','z'])
    #remove Hets and non-allowable atoms
    df = df[df['element'].isin(allowed_atoms)]
    df = df[df['hetero'].str.strip()=='']


    for chain_res, res_df in df.groupby(['chain', 'residue', 'resname']):
        # chain_res = res_df.index.values[0]
        # names.append('_'.join([str(x) for x in name]))
        chain, res, res_name = chain_res
        # only train on canonical residues
        if res_name not in res_label_dict:
            continue
        # sample each residue based on its frequency in train data
        if subsample:
            if not random.random() < res_wt_dict[res_name]:
                continue

        if not np.all([b in res_df['name'].to_list() for b in bb_atoms]):
            # print('residue missing atoms...   skipping')
            return
        CA_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

        CB_pos = CA_pos + (np.ones_like(CA_pos) * gly_CB_mu)

        # remove current residue from structure
        subunit_df = df[(df.chain != chain) | (df.residue != res)]
        # add backbone atoms back in
        res_bb = res_df[res_df['name'].isin(bb_atoms)]
        subunit_df = pd.concat([subunit_df, res_bb]).reset_index(drop=True)

        # environment = all atoms within 10*sqrt(3) angstroms (to enable a 20A cube)
        kd_tree = scipy.spatial.KDTree(subunit_df[['x','y','z']].to_numpy())
        subunit_pt_idx = kd_tree.query_ball_point(CB_pos, r=10.0*np.sqrt(3), p=2.0)

        sub_df = subunit_df.loc[subunit_pt_idx]
        tmp = sub_df.copy()
        tmp['subunit'] = '_'.join([str(x) for x in chain_res])
 
        subunits.append(tmp)
    return subunits

    

if __name__=="__main__":
    data_dir = O_DIR+'atom3d/data/residue_deletion'
    sharded_path = os.path.join(data_dir, 'test_pdbs@100/test_pdbs@100')
    out_env_path = os.path.join(data_dir, 'split/test_envs_unbalanced@100')
    
    # out_graph = os.path.join(data_dir, 'graph_pt')
    # generate_graph_data(sharded_path, out_graph, num_cores)
    shard_envs(sharded_path, out_env_path, num_threads=8, subsample=False)


