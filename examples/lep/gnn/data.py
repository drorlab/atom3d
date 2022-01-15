import numpy as np
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Batch, DataLoader
import atom3d.util.graph as gr

    
class GNNTransformLEP(object):
    def __init__(self, label_key):
        self.label_key = label_key
    
    def __call__(self, item):
        active = item['atoms_active']
        inactive = item['atoms_inactive']
        item['protein_active'] = active[active.chain != 'L']
        item['protein_inactive'] = inactive[inactive.chain != 'L']
        item['ligand_active'] = active[active.chain == 'L']
        item['ligand_inactive'] = inactive[inactive.chain == 'L']
        # transform protein and/or pocket to PTG graphs
        item = prot_graph_transform(item, atom_keys=['protein_active', 'protein_inactive'], label_key=self.label_key)
        item = mol_graph_transform(item, atom_key='ligand_active', label_key=self.label_key)
        item = mol_graph_transform(item, atom_key='ligand_inactive', label_key=self.label_key)
        
        node_feats, edges, edge_feats, node_pos = gr.combine_graphs(item['protein_active'], item['ligand_active'], edges_between=True)
        combined_active = Data(node_feats, edges, edge_feats, y=item[self.label_key], pos=node_pos)
        
        node_feats, edges, edge_feats, node_pos = gr.combine_graphs(item['protein_inactive'], item['ligand_inactive'], edges_between=True)
        combined_inactive = Data(node_feats, edges, edge_feats, y=item[self.label_key], pos=node_pos)
        
        return combined_active, combined_inactive

class CollaterLEP(object):
    """To be used with pre-computed graphs and atom3d.datasets.PTGDataset"""
    def __init__(self):
        pass
    def __call__(self, data_list):
        batch_1 = Batch.from_data_list([d[0] for d in data_list])
        batch_2 = Batch.from_data_list([d[1] for d in data_list])
        return batch_1, batch_2
    

        
if __name__=="__main__":
    save_dir = '/scratch/users/aderry/atom3d/lep_lig'
    data_dir = '/scratch/users/raphtown/atom3d_mirror/lmdb/LEP/splits/split-by-protein/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    transform = GNNTransformLEP(label_key='label')
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=transform)
    
    for i, item in enumerate(tqdm(train_dataset)):
        torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    for i, item in enumerate(tqdm(val_dataset)):
        torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))