import math
import os
import random

import dotenv as de
import numpy as np
import pandas as pd
import parallel as par
from tqdm import tqdm

de.load_dotenv(de.find_dotenv())

import atom3d.shard.shard as sh
import atom3d.torch.graph as gr


import torch
from torch_geometric.data import Data, Batch, Dataset
from torch.utils import data

seed = 131313
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


class LAP_Dataset_PTG(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(LAP_Dataset_PTG, self).__init__(root, transform, pre_transform)


    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return ['data_1.pt']
        

    def len(self):
        return len(os.listdir(self.processed_dir))

    def get(self, idx):
        batch = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        graph1, graph2 = batch.to_data_list()
        return graph1, graph2

class LAP_Dataset(data.IterableDataset):
    def __init__(self, sharded, seed=131313):
        self.sharded = sh.Sharded.load(sharded)
        self.num_shards = self.sharded.get_num_shards()
        self.seed = seed
        

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = dataset_generator(self.sharded, range(self.num_shards), 
                      shuffle=True)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.num_shards / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_shards)
            gen = dataset_generator(self.sharded, range(self.num_shards)[iter_start:iter_end],
                      shuffle=True)
        return gen


# class Collater(object):
#     def __init__(self, follow_batch):
#         self.follow_batch = follow_batch

#     def collate(self, data_list):
#         batch_1 = Batch.from_data_list([d[0] for d in data_list])
#         batch_2 = Batch.from_data_list([d[1] for d in data_list])
#         return batch_1, batch_2

#     def __call__(self, batch):
#         return self.collate(batch)

def custom_collate(data_list):
    batch_1 = Batch.from_data_list([d[0] for d in data_list])
    batch_2 = Batch.from_data_list([d[1] for d in data_list])
    return batch_1, batch_2


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
                             collate_fn=custom_collate, **kwargs)



def df_to_graph(struct_df, label):
    """
    struct_df: Dataframe
    """
    lig_df = struct_df[struct_df.chain == 'L']
    lig_graph = gr.prot_df_to_graph(lig_df)
    prot_df = struct_df[struct_df.chain != 'L']
    prot_graph = gr.prot_df_to_graph(prot_df)
    node_feats, edge_index, edge_feats, pos = gr.combine_graphs(prot_graph, lig_graph, edges_between=True)
    data = Data(node_feats, edge_index, edge_feats, y=label, pos=pos)

    return data


def dataset_generator(sharded, shard_indices, shuffle=True):
    """
    Generator that convert sharded HDF dataset to graphs
    """
    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)
        labels = sharded.read_shard(shard_idx, 'labels')

        if shuffle:
            groups = [df for _, df in shard.groupby('ensemble')]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for i, (target_name, target_df) in enumerate(shard.groupby(['ensemble'])):
           
            label = labels[(labels.ensemble == target_name)]['label'].values
            label = torch.FloatTensor(label == 'A')

            subunits = target_df.subunit.unique()

            for j, sub in enumerate(subunits):
                struct_df = target_df[target_df.subunit == sub]
                
                if '_active' in sub:
                    active_graph = df_to_graph(struct_df, label)
                    active_graph.name = f'{target_name}_active'
                elif '_inactive' in sub:
                    inactive_graph = df_to_graph(struct_df, label)
                    inactive_graph.name = f'{target_name}_inactive'

            graph_tuple = (active_graph, inactive_graph)
            yield graph_tuple

def save_graphs(sharded, out_dir, num_threads=8):
    num_shards = sharded.get_num_shards()
    inputs = [(sharded, shard_num, out_dir)
              for shard_num in range(num_shards)]

    # with multiprocessing.Pool(processes=num_threads) as pool:
    #     pool.starmap(_shard_envs, inputs)
    par.submit_jobs(_save_graphs, inputs, num_threads)
    _rename(out_dir)

def _rename(in_dir):
    in_files = os.listdir(in_dir)
    for i, f in tqdm(enumerate(in_files)):
        fpath = os.path.join(in_dir, f)
        outpath = os.path.join(in_dir, f'data_{i}.pt')
        os.rename(fpath, outpath)

def _save_graphs(sharded, shard_num, out_dir):
    print(f'Processing shard {shard_num:}')
    shard = sharded.read_shard(shard_num)
    labels = sharded.read_shard(shard_num, 'labels')
    curr_idx = 0
    for i, (target_name, target_df) in enumerate(shard.groupby(['ensemble'])):
           
        label = labels[(labels.ensemble == target_name)]['label'].values
        label = torch.FloatTensor(label == 'A')

        subunits = target_df.subunit.unique()

        for j, sub in enumerate(subunits):
            struct_df = target_df[target_df.subunit == sub]
            
            if '_active' in sub:
                active_graph = df_to_graph(struct_df, label)
                active_graph.name = f'{target_name}_active'
            elif '_inactive' in sub:
                inactive_graph = df_to_graph(struct_df, label)
                inactive_graph.name = f'{target_name}_inactive'


        pair = Batch.from_data_list([active_graph, inactive_graph])
        torch.save(pair, os.path.join(out_dir, f'data_{shard_num}_{curr_idx}.pt'))
        curr_idx += 1

if __name__ == "__main__":

    sharded = sh.Sharded.load(SC_DIR_R + 'atom3d/ligand_activity_prediction/split-20200528/pairs_val@10')

    if False:
        print('Testing LAP graph dataloader')
        # gen = dataset_generator(sharded, range(sharded.get_num_shards()), shuffle=True)

        dset = LAP_Dataset(SC_DIR_R+'atom3d/ligand_activity_prediction/split-20200528/pairs_train@10')
        loader = DataLoader(dset, batch_size=2, num_workers=8)

        for active_graph, inactive_graph in loader:
            # active_graph, inactive_graph = graph.to_data_list()
            print('Target {:}  -> nodes {:}/{:}, edges {:}/{:}, label {:}'.format(
                    active_graph.name, active_graph.num_nodes, inactive_graph.num_nodes, active_graph.num_edges, 
                    inactive_graph.num_edges, active_graph.y))

    graph_dir = SC_DIR+'atom3d/ligand_activity_prediction/graph_pt/val'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    save_graphs(sharded, graph_dir, num_threads=10)

