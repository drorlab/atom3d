import os
import subprocess
from tqdm import tqdm
import pdb

import numpy as np
import pandas as pd
import scipy.spatial
import random
import math
import parallel as par
import multiprocessing

import dotenv as de
de.load_dotenv(de.find_dotenv())

import sys
sys.path.append('../..')

import atom3d.util.datatypes as dt
import atom3d.util.shard as sh
import atom3d.util.graph as gr

import examples.cnn3d.subgrid_gen as subgrid_gen
import examples.cnn3d.util as util
from atom3d.residue_deletion.util import *

import torch
from torch_geometric.data import Data, Batch, Dataset
from torch.utils import data


class Resdel_Dataset_PTG(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Resdel_Dataset_PTG, self).__init__(root, transform, pre_transform)

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return ['data_1.pt']


    def len(self):
        return len(os.listdir(self.processed_dir))

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

class ResDel_Dataset(data.IterableDataset):
    def __init__(self, sharded, max_shards=None):
        self.sharded = sh.load_sharded(sharded)
        self.num_shards = self.sharded.get_num_shards()
        if max_shards:
            self.max_shards = max_shards
        else:
            self.max_shards = self.num_shards


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = dataset_generator(self.sharded, range(self.max_shards))

        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.max_shards / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.max_shards)
            gen = dataset_generator(self.sharded, range(self.max_shards)[iter_start:iter_end])
        return gen


class Collater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)

class DataLoader(data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch), **kwargs)


def df_to_graph(struct_df, chain_res, label):
    """
    label: residue label (int)
    chain_res: chain ID_residue ID_residue name defining center residue
    struct_df: Dataframe with entire environment
    grid_config: defined config
    """
    chain, resnum, _ = chain_res.split('_')
    ca_idx = np.where((struct_df.chain == chain) & (struct_df.residue == int(resnum)) & (struct_df.name == 'CA'))[0]

    node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(struct_df)
    data = Data(node_feats, edge_index, edge_feats, y=label, pos=pos)
    data.ca_idx = torch.LongTensor(ca_idx)
    data.n_nodes = data.num_nodes

    return data


def dataset_generator(sharded, shard_indices, shuffle=True, seed=131313):
    """
    Generate grids from sharded dataset
    """

    np.random.seed(seed)
    random.seed(seed)

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
            graph = df_to_graph(target_df, subunit, label)
            if graph is None:
                continue

            yield graph

def save_graphs(sharded, out_dir, num, num_threads=8):
    num_shards = sharded.get_num_shards()
    inputs = [(sharded, shard_num, out_dir)
              for shard_num in range(500, num)]

    # with multiprocessing.Pool(processes=num_threads) as pool:
    #     pool.starmap(_shard_envs, inputs)
    par.submit_jobs(_save_graphs, inputs, num_threads)
    _rename(out_dir)

def _rename(in_dir):
    curr_idx = 1765000
    in_files = os.listdir(in_dir)
    for i, f in tqdm(enumerate(in_files)):
        fpath = os.path.join(in_dir, f)
        outpath = os.path.join(in_dir, f'data_{curr_idx}.pt')
        os.rename(fpath, outpath)
        curr_idx += 1

def _save_graphs(sharded, shard_num, out_dir):
    print(f'Processing shard {shard_num:}')
    shard = sharded.read_shard(shard_num)
    for i, (e, target_df) in enumerate(shard.groupby(['ensemble', 'subunit'])):

        subunit = e[1]
        res_name = subunit.split('_')[-1]
        label = res_label_dict[res_name]
        # print(res_name)
        graph = df_to_graph(target_df, subunit, label)
        if graph is None:
            continue
        torch.save(graph, os.path.join(out_dir, f'data_{shard_num}_{i}.pt'))

if __name__ == "__main__":
    sharded = sh.load_sharded(O_DIR+'atom3d/data/residue_deletion/split/train_envs@1000')

    if False:
        print('Testing residue deletion generator')
        gen = dataset_generator(
            sharded, range(sharded.get_num_shards()), shuffle=True)

        for i, data in enumerate(gen):
            print('Generating sample {:} -> nodes {:}, edges {:}, label {}, CA index {}'.format(
                i, data.num_nodes, data.num_edges, data.y, data.ca_idx))

    num_cores = multiprocessing.cpu_count()

    graph_dir = SC_DIR+'atom3d/graph_pt/train_2'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    save_graphs(sharded, graph_dir, num=1000, num_threads=num_cores)