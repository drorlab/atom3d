import math
import os
import random

import dotenv as de
import numpy as np
import pandas as pd
from tqdm import tqdm

de.load_dotenv(de.find_dotenv())

import atom3d.util.formats as dt
import atom3d.shard.shard as sh
import atom3d.torch.graph as gr


import torch
from torch_geometric.data import Data, Batch, Dataset
from torch.utils import data

seed = 131313
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


class PSP_Dataset_PTG(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PSP_Dataset_PTG, self).__init__(root, transform, pre_transform)


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


class PSP_Dataset(data.IterableDataset):
    def __init__(self, sharded, scores_dir, seed=131313):
        self.sharded = sh.Sharded.load(sharded)
        self.scores_dir = scores_dir
        self.num_shards = self.sharded.get_num_shards()
        self.seed = seed
        

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = dataset_generator(self.sharded, range(self.num_shards), self.scores_dir, score_type='gdt_ts',
                      shuffle=True, max_targets=None,
                      max_decoys=None, max_dist_threshold=150.0)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.num_shards / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_shards)
            gen = dataset_generator(self.sharded, range(self.num_shards)[iter_start:iter_end], self.scores_dir, score_type='gdt_ts',
                      shuffle=True, max_targets=None,
                      max_decoys=None, max_dist_threshold=150.0)
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


def read_scores(scores_dir, targets):
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
    

def df_to_graph(struct_df, label):
    """
    label: residue label (int)
    chain_res: (chain ID, residue ID) to index df
    struct_df: Dataframe with entire structure
    """
    label = torch.FloatTensor(label)
    node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(struct_df)
    data = Data(node_feats, edge_index, edge_feats, y=label, pos=pos)

    return data


def dataset_generator(sharded, shard_indices, scores_dir, score_type='gdt_ts',
                      shuffle=True, max_targets=None,
                      max_decoys=None, max_dist_threshold=150.0):
    """
    Generator that convert sharded HDF dataset to grid features and
    also return the score labels. Skip structure with max distance above
    <max_dist_threshold> if specified.
    """
    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)
        all_target_names = shard['ensemble'].unique()
        scores_df = read_scores(scores_dir, all_target_names)

        if shuffle:
            groups = [df for _, df in shard.groupby('ensemble')]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for i, (target_name, target_df) in enumerate(shard.groupby(['ensemble'])):

            decoy_names = target_df.subunit.unique()
            if shuffle:
                p = np.random.permutation(len(decoy_names))
                decoy_names = decoy_names[p]
            if max_decoys is not None:
                decoy_names = decoy_names[:max_decoys]

            for j, decoy_name in enumerate(decoy_names):
                struct_df = target_df[target_df.subunit == decoy_name]

                score = scores_df[(scores_df.target == target_name) & \
                                    (scores_df.decoy == decoy_name)][score_type].values
                graph = df_to_graph(struct_df, score)

                if graph is None:
                    continue
                graph.name = '{:}/{:}.pdb'.format(target_name, decoy_name)

                # print('Target {:} ({:}/{:}): decoy {:} ({:}/{:}) -> nodes {:}, edges {:}, score {:}'.format(
                #     target_name, i+1, len(all_target_names), decoy_name, j+1,
                #     len(decoy_names),graph.num_nodes, graph.num_edges, graph.y))

                yield graph

def save_graphs(sharded, out_dir, num_threads=8):
    num_shards = sharded.get_num_shards()
    inputs = [(sharded, shard_num, out_dir)
              for shard_num in range(num_shards)]

    # with multiprocessing.Pool(processes=num_threads) as pool:
    #     pool.starmap(_shard_envs, inputs)
    # par.submit_jobs(_save_graphs, inputs, num_threads)
    _rename(out_dir)

def _rename(in_dir):
    for i, f in tqdm(enumerate(os.listdir(in_dir))):
        fpath = os.path.join(in_dir, f)
        outpath = os.path.join(in_dir, f'data_{i}.pt')
        os.rename(fpath, outpath)

def _save_graphs(sharded, shard_num, out_dir):
    print(f'Processing shard {shard_num:}')
    shard = sharded.read_shard(shard_num)
    all_target_names = shard['ensemble'].unique()
    scores_df = read_scores(scores_dir, all_target_names)
    curr_idx = 0
    for i, (target_name, target_df) in enumerate(shard.groupby(['ensemble'])):
        
        decoy_names = target_df.subunit.unique()

        for j, decoy_name in enumerate(decoy_names):
            struct_df = target_df[target_df.subunit == decoy_name]

            score = scores_df[(scores_df.target == target_name) & \
                                    (scores_df.decoy == decoy_name)]['gdt_ts'].values
            # print(res_name)
            graph = df_to_graph(struct_df, score)
            if graph is None:
                continue
            graph.name = '{:}/{:}.pdb'.format(target_name, decoy_name)
            torch.save(graph, os.path.join(out_dir, f'data_{shard_num}_{curr_idx}.pt'))
            curr_idx += 1

if __name__ == "__main__":
    sharded = sh.Sharded.load(SC_DIR_R + 'atom3d/protein_structure_prediction/casp/split_hdf/decoy_50/test_decoy_all@85')
    scores_dir = SC_DIR_R+'atom3d/protein_structure_prediction/casp/labels/scores'


    if False:
        print('Testing PSP graph generator')
        gen = dataset_generator(
            sharded, range(sharded.get_num_shards()), scores_dir, score_type='gdt_ts', shuffle=True, max_targets=10, max_decoys=10, max_dist_threshold=150.0)

        for i, graph in enumerate(gen):
            print('Generating graph {:} {:} -> nodes {:}, edges {:}, score {:}'.format(
                i, graph.name, graph.num_nodes, graph.num_edges, graph.y))

    graph_dir = O_DIR+'atom3d/data/psp/graph_pt/test'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    save_graphs(sharded, graph_dir, num_threads=8)

