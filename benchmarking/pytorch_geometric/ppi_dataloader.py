import math
import os
import random

import dotenv as de
import numpy as np
import pandas as pd
import parallel as par
import scipy.spatial
from tqdm import tqdm

de.load_dotenv(de.find_dotenv())

import atom3d.shard.shard as sh
import atom3d.torch.graph as gr
import atom3d.datasets.ppi.neighbors as nb


import torch
from torch_geometric.data import Data, Batch, Dataset
from torch.utils import data

seed = 131313
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

allowed_atoms = ['C', 'O', 'N', 'S']

class PPI_Dataset_PTG(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PPI_Dataset_PTG, self).__init__(root, transform, pre_transform)

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

class PPI_Dataset(data.IterableDataset):
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


def create_labels(positives, negatives, num_pos, neg_pos_ratio):
    frac = min(1, num_pos / positives.shape[0])
    positives = positives.sample(frac=frac)
    n = positives.shape[0] * neg_pos_ratio
    negatives = negatives.sample(n, random_state=seed, axis=0)
    labels = pd.concat([positives, negatives])[['chain0', 'residue0', 'chain1', 'residue1', 'label']]
    return labels

def remove_waters(df):
    # df = df[df['element'].isin(allowed_atoms)]
    df = df[df['hetero'].str.strip()!='W']
    return df

def df_to_graph(struct_df, chain_res, label):
    """
    struct_df: Dataframe
    """

    chain, resnum = chain_res
    res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue == resnum)]
    if 'CA' not in res_df.name.tolist():
        return None
    CA_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

    kd_tree = scipy.spatial.KDTree(struct_df[['x','y','z']].to_numpy())
    graph_pt_idx = kd_tree.query_ball_point(CA_pos, r=30.0, p=2.0)
    graph_df = struct_df.iloc[graph_pt_idx].reset_index(drop=True)
    ca_idx = np.where((graph_df.chain == chain) & (graph_df.residue == resnum) & (graph_df.name == 'CA'))[0]

    node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(graph_df)
    data = Data(node_feats, edge_index, edge_feats, y=label, pos=pos)
    data.ca_idx = torch.LongTensor(ca_idx)
    data.n_nodes = data.num_nodes

    return data


def dataset_generator(sharded, shard_indices, shuffle=True):
    """
    Generator that convert sharded HDF dataset to graphs
    """
    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)

        neighbors = sharded.read_shard(shard_idx, 'neighbors')

        if shuffle:
            groups = [df for _, df in shard.groupby('ensemble')]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for i, (ensemble_name, target_df) in enumerate(shard.groupby(['ensemble'])):

            sub_names, (bound1, bound2, _, _) = nb.get_subunits(target_df)
            # bound1 = remove_waters(bound1)
            # bound2 = remove_waters(bound2)
            positives = neighbors[neighbors.ensemble0 == ensemble_name]
            negatives = nb.get_negatives(positives, bound1, bound2)
            negatives['label'] = 0
            labels = create_labels(positives, negatives, neg_pos_ratio=1)
            labels = labels.sample(frac=1)
            
            for index, row in labels.iterrows():
                label = float(row['label'])
                chain_res1 = row[['chain0', 'residue0']].values
                chain_res2 = row[['chain1', 'residue1']].values
                graph1 = df_to_graph(bound1, chain_res1, label)
                if graph1 is None:
                    continue
                graph2 = df_to_graph(bound2, chain_res2, label)
                if graph2 is None:
                    continue
                yield graph1, graph2

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
    neighbors = sharded.read_shard(shard_num, 'neighbors')

    curr_idx = 0
    for i, (ensemble_name, target_df) in enumerate(shard.groupby(['ensemble'])):

        sub_names, (bound1, bound2, _, _) = nb.get_subunits(target_df)
        positives = neighbors[neighbors.ensemble0 == ensemble_name]
        negatives = nb.get_negatives(positives, bound1, bound2)
        negatives['label'] = 0
        labels = create_labels(positives, negatives, num_pos=10, neg_pos_ratio=1)
        
        for index, row in labels.iterrows():
            label = float(row['label'])
            chain_res1 = row[['chain0', 'residue0']].values
            chain_res2 = row[['chain1', 'residue1']].values
            graph1 = df_to_graph(bound1, chain_res1, label)
            graph2 = df_to_graph(bound2, chain_res2, label)
            if (graph1 is None) or (graph2 is None):
                continue

            pair = Batch.from_data_list([graph1, graph2])
            torch.save(pair, os.path.join(out_dir, f'data_{shard_num}_{curr_idx}.pt'))
            curr_idx += 1


if __name__ == "__main__":
    # split = sys.argv[1]
    db5_sharded = sh.Sharded.load(f'{os.environ["SC_DIR_R"]}atom3d/protein_interface_prediction/DB5/sharded/pairs@10')

    if False:
        print('Testing PPI graph dataloader')
        # gen = dataset_generator(sharded, range(sharded.get_num_shards()), shuffle=True)
        dset = PPI_Dataset(f'{os.environ["SC_DIR_R"]}atom3d/protein_interface_prediction/DIPS/split/pairs_pruned_val@1000')
        loader = DataLoader(dset, batch_size=2, num_workers=1)

        for i, (graph1, graph2) in enumerate(loader):
            # graph1, graph2 = graph.to_data_list()
            print('Target {:}  -> nodes {:}/{:}, edges {:}/{:}, label {:}'.format(
                    i, graph1.num_nodes, graph2.num_nodes, graph1.num_edges, 
                    graph2.num_edges, graph1.y))

    graph_dir = f'{os.environ["SC_DIR"]}atom3d/protein_interface_prediction/graph_pt/db5'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    save_graphs(db5_sharded, graph_dir, num_threads=10)

