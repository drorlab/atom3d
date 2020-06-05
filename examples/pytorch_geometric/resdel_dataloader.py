import os
import subprocess
import tqdm
import pdb

import numpy as np
import pandas as pd
import scipy.spatial

import dotenv as de
de.load_dotenv(de.find_dotenv())

import sys
sys.path.append('../..')

import atom3d.util.datatypes as dt
import atom3d.util.shard as sh
import atom3d.util.graph as gr

import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util

from torch_geometric.data import Data

label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}
res_label_dict={'HIS':0,'LYS':1,'ARG':2,'ASP':3,'GLU':4,'SER':5,'THR':6,'ASN':7,'GLN':8,'ALA':9,'VAL':10,'LEU':11,'ILE':12,'MET':13,'PHE':14,'TYR':15,'TRP':16,'PRO':17,'GLY':18,'CYS':19}
bb_atoms = ['N', 'CA', 'C']

gly_CB_mu = np.array([-0.5311191 , -0.75842446,  1.2198311 ], dtype=np.float32)
gly_CB_sigma = np.array([[1.63731114e-03, 2.40018381e-04, 6.38361679e-04],
       [2.40018381e-04, 6.87853419e-05, 1.43898267e-04],
       [6.38361679e-04, 1.43898267e-04, 3.25022011e-04]], dtype=np.float32)


def df_to_graph(struct_df, chain_res, label):
    """
    label: residue label (int)
    chain_res: (chain ID, residue ID) to index df
    struct_df: Dataframe with entire structure
    grid_config: defined config
    """
    res_df = struct_df.loc[chain_res]
    if not np.all([b in res_df['name'].to_list() for b in bb_atoms]):
        # print('residue missing atoms... skipping')
        return
    CA_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]
    # N_pos = res_df[res_df['name']=='N'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]
    # C_pos = res_df[res_df['name']=='C'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

    CB_pos = CA_pos + (np.ones_like(CA_pos) * gly_CB_mu)

    # remove current residue from structure
    struct_df = struct_df.drop(index=chain_res)

    kd_tree = scipy.spatial.KDTree(struct_df[['x','y','z']].to_numpy())
    graph_pt_idx = kd_tree.query_ball_point(CB_pos, r=10.0, p=2.0)
    graph_df = struct_df.iloc[graph_pt_idx]

    node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(graph_df)
    data = Data(node_feats, edge_index, edge_feats, y=label, pos=pos)

    return data


def dataset_generator(sharded, shard_indices, shuffle=True):
    """
    Generate grids from sharded dataset
    """
    
    # if shuffle:
    #     p = np.random.permutation(len(all_target_names))
    #     all_target_names = all_target_names[p]

    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)
        shard = shard[shard['hetero'].str.strip()=='']

        for e, target_df in shard.groupby(['ensemble']):
            target_df = target_df.set_index(['chain', 'residue', 'resname'])

            #shuffle residues
            residues = np.unique(target_df.index.values)
            if shuffle:
                np.random.shuffle(residues)

            for i, res_df in target_df.groupby(['chain', 'residue', 'resname']):

                chain_res = res_df.index.values[0]
                res_name = chain_res[-1]
                # only train on canonical residues
                if res_name not in res_label_dict:
                    continue
                label = res_label_dict[res_name]
                # print(res_name)
                graph = df_to_graph(target_df, chain_res, label)
                if graph is None:
                    continue

                yield graph


if __name__ == "__main__":
    sharded = sh.load_sharded('/oak/stanford/groups/rbaltman/aderry/atom3d/data/residue_deletion/test_pdbs@100/test_pdbs@100')

    print('Testing residue deletion generator')
    gen = dataset_generator(
        sharded, shuffle=True)

    for i, data in enumerate(gen):
        print('Generating sample {:} -> nodes {:}, edges {:}'.format(
            i, data.num_nodes, data.num_edges))

