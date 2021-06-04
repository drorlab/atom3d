import numpy as np
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import prot_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, Batch
from torch.utils.data import DataLoader
import atom3d.util.graph as gr

    
class GNNTransformMSP(object):
    def __init__(self):
        pass
    
    def __call__(self, item):
        # transform each atoms df to PTG graphs
        mutation = item['id'].split('_')[-1]
        orig_df = item['original_atoms'].reset_index(drop=True)
        mut_df = item['mutated_atoms'].reset_index(drop=True)
        orig_idx = self._extract_mut_idx(orig_df, mutation)
        mut_idx = self._extract_mut_idx(mut_df, mutation)

        item = prot_graph_transform(item, atom_keys=['original_atoms', 'mutated_atoms'], label_key='label')
        orig_graph = self._augment_graph(item['original_atoms'], orig_idx)
        mut_graph = self._augment_graph(item['mutated_atoms'], mut_idx)
        return orig_graph, mut_graph
    
    def _extract_mut_idx(self, df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[(df.chain.values == chain) & (df.residue.values == res)].values
        return torch.LongTensor(idx)
    
    def _augment_graph(self, graph, idx):
        graph.mut_idx = idx
        graph.num_mut_atoms = len(idx)
        graph.n_nodes = graph.num_nodes
        graph.y = torch.FloatTensor([int(x) for x in graph.y])
        return graph
    

class CollaterMSP(object):
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
        total_n_mut = 0
        
        for i in range(bs-1):
            n_nodes = graph.n_nodes[i].item()
            total_n += n_nodes
            total_n_mut += graph.num_mut_atoms[i]
            graph.mut_idx[total_n_mut:total_n_mut+graph.num_mut_atoms[i+1]] += total_n
        return graph

    def __call__(self, batch):
        return self.collate(batch)

        
if __name__=="__main__":
    from tqdm import tqdm

    save_dir = '/scratch/users/aderry/atom3d/msp'
    data_dir = '/scratch/users/raphtown/atom3d_mirror/lmdb/MSP/splits/split-by-sequence-identity-30/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    transform = GNNTransformMSP()
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=transform)
    
    # train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=4, collate_fn=CollaterMSP(batch_size=1))
    # val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=4, collate_fn=CollaterMSP(batch_size=1))
    # test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, collate_fn=CollaterMSP(batch_size=1))
    # for item in dataset[0]:
    #     print(item, type(dataset[0][item]))
    
    # for i, item in enumerate(tqdm(train_dataset)):
    #     torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    # print('processing validation dataset...')
    # for i, item in enumerate(tqdm(val_dataset)):
    #     torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    print('processing test dataset...')
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))