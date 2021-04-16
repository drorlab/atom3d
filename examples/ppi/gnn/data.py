import numpy as np
import pandas as pd
import os
import torch
import random
import math
import scipy.spatial

from atom3d.util.transforms import prot_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, Batch
from torch.utils.data import IterableDataset, DataLoader
import atom3d.util.graph as gr
import atom3d.datasets.ppi.neighbors as nb


class PPIDataset(IterableDataset):
    def __init__(self, lmdb_path):
        self.dataset = LMDBDataset(lmdb_path)
        

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = dataset_generator(self.dataset, range(len(self.dataset)), 
                      shuffle=True)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
            gen = dataset_generator(self.dataset, range(len(self.dataset))[iter_start:iter_end],
                      shuffle=True)
        return gen

class CollaterPPI(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def collate(self, data_list):
        if len(data_list) != self.batch_size:
            bs = len(data_list)
        else:
            bs = self.batch_size
        batch_1 = self.adjust_graph_indices(Batch.from_data_list([d[0] for d in data_list]), bs)
        batch_2 = self.adjust_graph_indices(Batch.from_data_list([d[1] for d in data_list]), bs)
        return batch_1, batch_2
    
    def adjust_graph_indices(self, graph, bs):
        total_n = 0
        for i in range(bs-1):
            n_nodes = graph.n_nodes[i].item()
            total_n += n_nodes
            graph.ca_idx[i+1] += total_n
        return graph

    def __call__(self, batch):
        return self.collate(batch)

def df_to_graph(struct_df, chain_res, label):
    """
    Extracts atoms within 30A of CA atom and computes graph
    """

    chain, resnum = chain_res
    res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue == resnum)]
    if 'CA' not in res_df.name.tolist():
        return None
    ca_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

    kd_tree = scipy.spatial.KDTree(struct_df[['x','y','z']].to_numpy())
    graph_pt_idx = kd_tree.query_ball_point(ca_pos, r=30.0, p=2.0)
    graph_df = struct_df.iloc[graph_pt_idx].reset_index(drop=True)
    ca_idx = np.where((graph_df.chain == chain) & (graph_df.residue == resnum) & (graph_df.name == 'CA'))[0]
    if len(ca_idx) > 0:
        return None

    node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(graph_df)
    data = Data(node_feats, edge_index, edge_feats, y=label, pos=pos)
    data.ca_idx = torch.LongTensor(ca_idx)
    data.n_nodes = data.num_nodes

    return data

def create_labels(positives, negatives, num_pos, neg_pos_ratio):
    frac = min(1, num_pos / positives.shape[0])
    positives = positives.sample(frac=frac)
    n = positives.shape[0] * neg_pos_ratio
    negatives = negatives.sample(n, random_state=0, axis=0)
    labels = pd.concat([positives, negatives])[['chain0', 'residue0', 'chain1', 'residue1', 'label']]
    return labels

def remove_waters(df):
    # df = df[df['element'].isin(allowed_atoms)]
    df = df[df['hetero'].str.strip()!='W']
    return df

def dataset_generator(dataset, indices, shuffle=True):
    """
    Generator that convert sharded HDF dataset to graphs
    """
    for idx in indices:
        data = dataset[idx]

        neighbors = data['atoms_neighbors']
        pairs = data['atoms_pairs']

        if shuffle:
            groups = [df for _, df in pairs.groupby('ensemble')]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for i, (ensemble_name, target_df) in enumerate(pairs.groupby(['ensemble'])):

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
                yield graph1, graph2

if __name__=="__main__":
    from tqdm import tqdm
        
    dataset = PPIDataset(os.path.join('/scratch/users/raphtown/atom3d_mirror/lmdb/PPI/splits/DIPS-split/data', 'train'))
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=CollaterPPI(batch_size=3), num_workers=4)
    for graph1, graph2 in dataloader:
        print(graph1)
        print(graph2)
        break