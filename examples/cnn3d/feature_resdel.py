import os
import subprocess
from tqdm import tqdm
import parallel as par
import multiprocessing

import numpy as np
import pandas as pd
import random
import torch

import dotenv as de
de.load_dotenv(de.find_dotenv(usecwd=True))

import sys
sys.path.append('../..')

import atom3d.util.datatypes as dt
import atom3d.util.shard as sh
from atom3d.residue_deletion.util import *

import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
    chain, resnum, _ = chain_res.split('_')
    res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue == int(resnum))]

    CA_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]
    N_pos = res_df[res_df['name']=='N'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]
    C_pos = res_df[res_df['name']=='C'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

    CB_pos = CA_pos + (np.ones_like(CA_pos) * gly_CB_mu)

    rot_mat = get_rot_matrix(CA_pos, N_pos, C_pos, CB_pos)

    grid = subgrid_gen.get_voxels(struct_df, CB_pos, config=grid_config, rot_mat=rot_mat)
    return grid.transpose(3, 0, 1, 2)


def dataset_generator(sharded, shard_indices, grid_config=grid_config, shuffle=True):
    """
    Generate grids from sharded dataset
    """

    np.random.seed(grid_config.random_seed)
    random.seed(grid_config.random_seed)

    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)
        if shuffle:
            groups = [df for _, df in shard.groupby(['ensemble', 'subunit'])]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for e, target_df in shard.groupby(['ensemble', 'subunit']):
            subunit = e[1]
            res_name = subunit.split('_')[-1]
            label = res_label_dict[res_name]
            # print(res_name)
            feature = df_to_feature(target_df, subunit, grid_config)
            if feature is None:
                continue

            yield feature, label

def save_graphs(sharded, out_dir, num, num_threads=8):
    num_shards = sharded.get_num_shards()
    inputs = [(sharded, shard_num, out_dir)
              for shard_num in range(num)]

    # with multiprocessing.Pool(processes=num_threads) as pool:
    #     pool.starmap(_shard_envs, inputs)
    # par.submit_jobs(_save_graphs, inputs, num_threads)
    _rename(out_dir)

def _rename(in_dir):
    import glob
    curr_idx = 0 #3530580
    in_files = glob.glob(os.path.join(in_dir, 'label_*.pt'))
    # in_files = os.listdir(in_dir)
    print([os.path.basename(f) for f in in_files[:100]])
    return
    
    for f in tqdm(in_files):
        fpath = os.path.join(in_dir, f)
        outpath = os.path.join(in_dir, f'data_{curr_idx}.pt')
        os.rename(fpath, outpath)
        root = os.path.basename(f)[5:]
        label_file = os.path.join(in_dir, f'label_{root}')
        label_outpath = os.path.join(in_dir, f'label_{curr_idx}.pt')
        os.rename(label_file, label_outpath)
        curr_idx += 1


def _save_graphs(sharded, shard_num, out_dir):
    print(f'Processing shard {shard_num:}')
    shard = sharded.read_shard(shard_num)
    curr_idx = 0
    for e, target_df in shard.groupby(['ensemble', 'subunit']):
        subunit = e[1]
        res_name = subunit.split('_')[-1]
        label = res_label_dict[res_name]
        # print(res_name)
        feature = df_to_feature(target_df, subunit, grid_config)
        if feature is None:
            continue

        torch.save(feature, os.path.join(out_dir, f'data_{shard_num}_{curr_idx}.pt'))
        torch.save(label, os.path.join(out_dir, f'label_{shard_num}_{curr_idx}.pt'))
        curr_idx+=1


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
    for shard in tqdm(sharded.iter_shards()):
        for e, target_df in shard.groupby(['ensemble', 'subunit']):
            subunit = e[1]
            res_name = subunit.split('_')[-1]
            cts[res_name] += 1
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

def get_residue_weights():
    res_wt_dict={}
    with open('train_res_weights') as f:
        for line in f:
            res, wt = line.strip().split()
            res_wt_dict[res] = float(wt)
    return res_wt_dict


if __name__ == "__main__":
<<<<<<< HEAD
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
=======
    sharded = sh.load_sharded(O_DIR + 'atom3d/data/residue_deletion/split/train_envs@1000')

    if False:
        cts, pcts = get_all_labels(sharded)
        print(pcts)
        
        total_ct = 0
        for res, ct in cts.items():
            total_ct += int(ct)
                
        print(total_ct, 'total residues')
        # print('Testing residue deletion generator')
        # gen = dataset_generator(sharded, range(sharded.get_num_shards()), grid_config=grid_config)
        gen = dataset_generator(sharded, range(sharded.get_num_shards()), grid_config=grid_config)

        for i, (feature, label) in tqdm(enumerate(gen)):
            print('Generating sample {:} -> feature {:}, label {:}'.format(
                i, feature.shape, label))
        #     plot_cube(feature.transpose(1,2,3,0)[:,:,:,0], f'test_car_{label}.png')
        #     if i == 9:
        #         break
    num_cores = multiprocessing.cpu_count()

    graph_dir = SC_DIR + 'atom3d/residue_deletion/cube_pt/train'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    save_graphs(sharded, graph_dir, num=1000, num_threads=num_cores)

>>>>>>> GNN training code
