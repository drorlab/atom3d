import numpy as np
import os
import torch
from tqdm import tqdm
from atom3d.util.transforms import prot_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Batch, DataLoader
import atom3d.util.graph as gr

    
class GNNTransformLEP(object):
    def __init__(self, atom_keys, label_key):
        self.atom_keys = atom_keys
        self.label_key = label_key
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        item = prot_graph_transform(item, atom_keys=self.atom_keys, label_key=self.label_key)
        
        return item

class CollaterLEP(object):
    """To be used with pre-computed graphs and atom3d.datasets.PTGDataset"""
    def __init__(self):
        pass
    def __call__(self, data_list):
        batch_1 = Batch.from_data_list([d[0] for d in data_list])
        batch_2 = Batch.from_data_list([d[1] for d in data_list])
        return batch_1, batch_2
    

        
if __name__=="__main__":
    save_dir = '/scratch/users/aderry/atom3d/lep'
    data_dir = '/scratch/users/raphtown/atom3d_mirror/lmdb/LEP/splits/split-by-protein/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    transform = PairedGraphTransform('atoms_active', 'atoms_inactive', label_key='label')
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=transform)
    
    # train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4)
    # for item in dataset[0]:
    #     print(item, type(dataset[0][item]))
    for i, item in enumerate(tqdm(train_dataset)):
        torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    for i, item in enumerate(tqdm(val_dataset)):
        torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))