import numpy as np
import os
import sys
from tqdm import tqdm
import torch
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader
import atom3d.util.graph as gr

    
class GNNTransformLBA(object):
    def __init__(self, pocket_only=True):
        self.pocket_only = pocket_only
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        if self.pocket_only:
            item = prot_graph_transform(item, atom_keys=['atoms_pocket'], label_key='scores')
        else:
            item = prot_graph_transform(item, atom_keys=['atoms_protein', 'atoms_pocket'], label_key='scores')
        # transform ligand into PTG graph
        item = mol_graph_transform(item, 'atoms_ligand', 'scores', use_bonds=True, onehot_edges=False)
        node_feats, edges, edge_feats, node_pos = gr.combine_graphs(item['atoms_pocket'], item['atoms_ligand'], edges_between=True)
        combined_graph = Data(node_feats, edges, edge_feats, y=item['scores']['neglog_aff'], pos=node_pos)
        return combined_graph
    

        
if __name__=="__main__":
    seqid = sys.argv[1]
    save_dir = '/scratch/users/aderry/atom3d/lba_ptg_' + str(seqid)
    data_dir = f'/scratch/users/raphtown/atom3d_mirror/lmdb/LBA/splits/split-by-sequence-identity-{seqid}/data'
    # data_dir = '/scratch/users/aderry/atom3d/lba_30_withH/split/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=GNNTransformLBA())
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=GNNTransformLBA())
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=GNNTransformLBA())
    
    print('processing train dataset...')
    for i, item in enumerate(tqdm(train_dataset)):
        torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    print('processing validation dataset...')
    for i, item in enumerate(tqdm(val_dataset)):
        torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    print('processing test dataset...')
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))
