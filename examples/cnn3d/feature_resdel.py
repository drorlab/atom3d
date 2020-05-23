import os
import subprocess
from tqdm import tqdm
import pdb

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn

import dotenv as de
de.load_dotenv(de.find_dotenv())

import sys
sys.path.append('../..')

import atom3d.util.datatypes as dt
import atom3d.util.shard as sh

import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}
res_label_dict={'HIS':0,'LYS':1,'ARG':2,'ASP':3,'GLU':4,'SER':5,'THR':6,'ASN':7,'GLN':8,'ALA':9,'VAL':10,'LEU':11,'ILE':12,'MET':13,'PHE':14,'TYR':15,'TRP':16,'PRO':17,'GLY':18,'CYS':19}
bb_atoms = ['N', 'CA', 'C', 'O']

gly_CB_mu = np.array([-0.5311191 , -0.75842446,  1.2198311 ], dtype=np.float32)
gly_CB_sigma = np.array([[1.63731114e-03, 2.40018381e-04, 6.38361679e-04],
       [2.40018381e-04, 6.87853419e-05, 1.43898267e-04],
       [6.38361679e-04, 1.43898267e-04, 3.25022011e-04]], dtype=np.float32)

grid_config = util.dotdict({
    # Mapping from elements to position in channel dimension.
    'element_mapping': {
        'C': 0,
        'O': 1,
        'N': 2,
        'S': 3,
        'P': 4
    },
    # Radius of the grids to generate, in angstroms.
    'radius': 9.5,
    # Resolution of each voxel, in angstroms.
    'resolution': 1.0,
    # Number of directions to apply for data augmentation.
    # 'num_directions': 20,
    # Number of rolls to apply for data augmentation.
    # 'num_rolls': 20,
    # Random seed
    'random_seed': 131313,
})

def get_rot_matrix(CA_pos, N_pos, C_pos, CB_pos):
 
    axis_x = np.array(N_pos) - np.array(CA_pos)  
    pseudo_axis_y = np.array(C_pos) - np.array(CA_pos)  
    axis_z = np.cross(axis_x , pseudo_axis_y)

    direction = np.array(CB_pos) - np.array(CA_pos) 
    axis_z *= np.sign( direction.dot(axis_z.T) ) 
    axis_y= np.cross(axis_z, axis_x)

    axis_x/=np.sqrt(sum(axis_x**2))
    axis_y/=np.sqrt(sum(axis_y**2))
    axis_z/=np.sqrt(sum(axis_z**2))

    transform=np.array([axis_x, axis_y, axis_z]).T
    return transform


def df_to_feature(struct_df, chain_res, grid_config):
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
    N_pos = res_df[res_df['name']=='N'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]
    C_pos = res_df[res_df['name']=='C'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

    # if label == 'GLY':
    #     #sample CB position
    #     CB_pos = mvn.rvs(mean=gly_CB_mu, cov=gly_CB_sigma)
    # else:
    # try:
        # CB_pos = res_df[res_df['name']=='CB'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]
    # except IndexError:
    CB_pos = CA_pos + (np.ones_like(CA_pos) * gly_CB_mu)

    rot_mat = get_rot_matrix(CA_pos, N_pos, C_pos, CB_pos)

    # remove current residue from structure
    struct_df = struct_df.drop(index=chain_res)
    # add backbone atoms back in
    res_bb = res_df['name'].isin(bb_atoms)
    struct_df = pd.concat([struct_df, res_bb])

    grid = subgrid_gen.get_voxels(struct_df, CB_pos, config=grid_config, rot_mat=rot_mat)
    return grid.transpose(3, 0, 1, 2)


def dataset_generator(sharded, shard_indices, grid_config=grid_config, shuffle=True):
    """
    Generate grids from sharded dataset
    """

    # np.random.seed(grid_config.random_seed)
    
    # if shuffle:
    #     p = np.random.permutation(len(all_target_names))
    #     all_target_names = all_target_names[p]

    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)
        shard = shard[shard['hetero'].str.strip()=='']

        for e, target_df in shard.groupby(['ensemble']):
            target_df = target_df.set_index(['chain', 'residue', 'resname'])

            # #shuffle residues
            # residues = np.unique(target_df.index.values)
            # if shuffle:
            #     np.random.shuffle(residues)

            for i, res_df in target_df.groupby(['chain', 'residue', 'resname']):

                chain_res = res_df.index.values[0]
                res_name = chain_res[-1]
                # only train on canonical residues
                if res_name not in res_label_dict:
                    continue
                label = res_label_dict[res_name]
                # print(res_name)
                feature = df_to_feature(target_df, chain_res, grid_config)
                if feature is None:
                    continue

                yield feature, label

def sample_generator(sharded, target_names, shuffle=True):
    """
    Generate grids from subset of sharded dataset, specified by target_names
    """
    # if shuffle:
    #     p = np.random.permutation(len(target_names))
    #     target_names = target_names.iloc[p]

    for i, target_name in enumerate(target_names):
        target_df = sh.read_ensemble(sharded, target_name)
        target_df = target_df[target_df['hetero'].str.strip()=='']
        target_df = target_df.set_index(['chain', 'residue'])

        residues = np.unique(target_df.index.values)
        if shuffle:
            np.random.shuffle(residues)


        for j, res in enumerate(residues):

            res_name = target_df.loc[res]['resname'].unique()[0]
            # only train on canonical residues
            if res_name not in res_label_dict:
                continue
            label = res_label_dict[res_name]
            # print(res_name)
            feature = df_to_feature(target_df, res_name, res, grid_config)
            if feature is None:
                continue

            yield feature, label

def normalize_plot(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, name='test.png'):
    cube = normalize_plot(cube)

    facecolors = cm.viridis(cube)
    facecolors[:,:,:,-1] = 0.1*cube
    facecolors = explode(facecolors)

    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x, y, z, filled, facecolors=facecolors, alpha=0.5)
    plt.savefig(name)

def get_all_labels(sharded):
    import json
    cts = {k:0 for k in res_label_dict.keys()}
    total = 0
    for shard in sharded.iter_shards():
        shard = shard.set_index(['chain', 'residue', 'resname'])
        residues = np.unique(shard.index.values)
        for r in residues:
            resname = r[2]
            if resname not in res_label_dict:
                continue
            cts[resname] += 1
            total += 1
    pcts = {k:0 for k in res_label_dict.keys()}
    min_ct = np.inf
    min_res = None
    with open('train_res_counts', 'w') as f:
        for res, ct in cts.items():
            f.write(f'{res}\t{ct}\n')
            pcts[res] = ct / float(total)
            if ct < min_ct:
                min_res = res
    with open('train_res_pcts', 'w') as f:
        for res, pct in pcts.items():
            f.write(f'{res}\t{pct}\n')
    with open('train_res_weights', 'w') as f:
        for res, ct in cts.items():
            f.write(f'{res}\t{float(cts[min_res])/ct}\n')
    return cts, pcts



if __name__ == "__main__":
    sharded = sh.load_sharded('/oak/stanford/groups/rbaltman/aderry/atom3d/data/residue_deletion/test_pdbs@100/test_pdbs@100')
    cts, pcts = get_all_labels(sharded)
    print(cts)
    print(pcts)
    quit()
    print('Testing residue deletion generator')
    gen = dataset_generator(
        sharded, range(sharded.get_num_shards()), grid_config, shuffle=True)

    for i, (feature, label) in enumerate(gen):
        print('Generating sample {:} -> feature {:}, label {:}'.format(
            i, feature.shape, label))
        plot_cube(feature.transpose(1,2,3,0)[:,:,:,0], f'test_car_{label}.png')
        if i == 9:
            break


